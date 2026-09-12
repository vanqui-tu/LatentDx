"""Differentiable fixed-route latent KV communication.

The protocol transports only a newly encoded ``(m + 2)`` position KV block:
``BEGIN``, ``m`` distilled positions, and ``END``.  KV blocks are tuples of
``(key, value)`` tensors, one pair per decoder layer, shaped
``[batch, heads, positions, head_dim]``.  Blocks are concatenated in stable
child/branch order before a relay re-encodes them with its private prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from ..hf_medlatent_h import _build_position_ids, _iter_key_value_pairs
from ..modules import BoundaryEmbeddings, LatentDistiller

KVBlock = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def _as_block(cache: Any) -> KVBlock:
    """Normalize a HF legacy/DynamicCache value without detaching tensors."""
    return tuple((key, value) for key, value in _iter_key_value_pairs(cache))


def merge_kv_blocks(blocks: Sequence[KVBlock]) -> KVBlock | None:
    """Concatenate child summaries along the sequence dimension."""
    if not blocks:
        return None
    normalized = [_as_block(block) for block in blocks]
    layer_count = len(normalized[0])
    if any(len(block) != layer_count for block in normalized):
        raise ValueError("all KV blocks must have the same number of layers")
    merged: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in range(layer_count):
        keys = [block[layer][0] for block in normalized]
        values = [block[layer][1] for block in normalized]
        if any(key.ndim != 4 or value.ndim != 4 for key, value in zip(keys, values)):
            raise ValueError("KV tensors must have shape [batch, heads, positions, head_dim]")
        if any(key.shape[:2] != keys[0].shape[:2] or key.shape[3] != keys[0].shape[3] for key in keys):
            raise ValueError("KV blocks have incompatible key shapes")
        if any(value.shape != key.shape for key, value in zip(keys, values)):
            raise ValueError("key and value shapes must match")
        merged.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
    return tuple(merged)


def slice_kv_block(cache: Any, num_positions: int) -> KVBlock:
    """Copy final positions without retaining full prompt-cache storage."""
    if num_positions <= 0:
        raise ValueError("num_positions must be positive")
    return tuple((key[:, :, -num_positions:, :].clone(), value[:, :, -num_positions:, :].clone())
                 for key, value in _iter_key_value_pairs(cache))


def kv_wire_bytes(block: KVBlock, *, metadata_bytes: int = 32) -> int:
    """Estimate serialized tensor cost for evaluation accounting."""
    if metadata_bytes < 0:
        raise ValueError("metadata_bytes must be non-negative")
    return int(metadata_bytes + sum(key.numel() * key.element_size() + value.numel() * value.element_size()
                                    for key, value in block))


class DistributedLatentProtocol:
    """Frozen-backbone KV protocol for a prescribed request/return route."""

    def __init__(self, model: torch.nn.Module, distiller: LatentDistiller,
                 boundary: BoundaryEmbeddings, *, num_latents: int = 8):
        if num_latents <= 0:
            raise ValueError("num_latents must be positive")
        self.model = model
        self.distiller = distiller
        self.boundary = boundary
        self.num_latents = int(num_latents)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _decoder(self, **kwargs):
        # CausalLM wrappers expose the decoder as base_model.  Keeping this
        # call separate makes tiny fake decoders usable in protocol tests.
        decoder = getattr(self.model, "base_model", self.model)
        return decoder(**kwargs)

    def _decoder_with_cache(self, *, input_ids: torch.Tensor | None = None,
                            inputs_embeds: torch.Tensor | None = None,
                            attention_mask: torch.Tensor, position_ids: torch.Tensor,
                            past_key_values: KVBlock | None):
        """Checkpoint one decoder call while retaining its differentiable KV output."""
        if not torch.is_grad_enabled():
            return self._decoder(
                input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values, use_cache=True, return_dict=True,
            )
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        flat_past = () if past_key_values is None else tuple(
            tensor for key, value in past_key_values for tensor in (key, value)
        )
        input_layer_count = len(flat_past) // 2

        def forward(embeddings: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor, *flat: torch.Tensor):
            past = None if not flat else tuple(
                (flat[2 * index], flat[2 * index + 1]) for index in range(input_layer_count)
            )
            output = self._decoder(
                inputs_embeds=embeddings, attention_mask=mask, position_ids=positions,
                past_key_values=past, use_cache=True, return_dict=True,
            )
            return (*tuple(tensor for key, value in _as_block(output.past_key_values) for tensor in (key, value)),
                    output.last_hidden_state)

        outputs = checkpoint(forward, inputs_embeds, attention_mask, position_ids, *flat_past, use_reentrant=False)
        *flat_cache, hidden = outputs
        layer_count = len(flat_cache) // 2
        return SimpleNamespace(
            past_key_values=tuple((flat_cache[2 * index], flat_cache[2 * index + 1]) for index in range(layer_count)),
            last_hidden_state=hidden,
        )

    def _append_embedding(self, embedding: torch.Tensor, prefix_mask: torch.Tensor, past):
        batch = prefix_mask.shape[0]
        dtype = self.model.get_input_embeddings().weight.dtype
        inputs = embedding.to(device=prefix_mask.device, dtype=dtype).view(1, 1, -1).expand(batch, 1, -1)
        attention = torch.cat([prefix_mask, prefix_mask.new_ones((batch, 1))], dim=1)
        output = self._decoder_with_cache(inputs_embeds=inputs, attention_mask=attention,
                                          position_ids=_build_position_ids(prefix_mask, 1), past_key_values=past)
        return output.past_key_values, attention

    def rollout(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                incoming: Sequence[KVBlock] = ()) -> KVBlock:
        """Re-encode local private evidence and incoming child summaries."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both be [batch, sequence]")
        incoming_block = merge_kv_blocks(tuple(incoming))
        past = incoming_block
        prefix_len = 0 if past is None else past[0][0].shape[2]
        prefix_mask = attention_mask.new_ones((input_ids.shape[0], prefix_len))
        if past is None:
            local_attention = attention_mask
        else:
            local_attention = torch.cat([prefix_mask, attention_mask], dim=1)
        output = self._decoder_with_cache(input_ids=input_ids, attention_mask=local_attention,
                                          position_ids=_build_position_ids(prefix_mask, input_ids.shape[1]),
                                          past_key_values=past)
        past = output.past_key_values
        # Decoder hidden states cover only the newly supplied local prompt;
        # incoming KV positions are present in the cache, not this tensor.
        lengths = attention_mask.sum(dim=1).long() - 1
        hidden = output.last_hidden_state[torch.arange(input_ids.shape[0], device=input_ids.device), lengths, :]
        prefix_mask = local_attention
        past, prefix_mask = self._append_embedding(self.boundary.begin, prefix_mask, past)
        for step in range(self.num_latents):
            latent = self.distiller.begin_embedding(dtype=self.model.get_input_embeddings().weight.dtype,
                                                    device=input_ids.device).expand(input_ids.shape[0], 1, -1)
            if step:
                latent = self.distiller(hidden).unsqueeze(1)
            attention = torch.cat([prefix_mask, prefix_mask.new_ones((input_ids.shape[0], 1))], dim=1)
            latent_out = self._decoder_with_cache(inputs_embeds=latent, attention_mask=attention,
                                                   position_ids=_build_position_ids(prefix_mask, 1),
                                                   past_key_values=past)
            past, prefix_mask = latent_out.past_key_values, attention
            hidden = latent_out.last_hidden_state[:, -1, :]
        past, _ = self._append_embedding(self.boundary.end, prefix_mask, past)
        return slice_kv_block(past, self.num_latents + 2)

    def source_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      branch_blocks: Sequence[KVBlock] = ()) -> torch.Tensor:
        """Return logits for a source prompt conditioned on final branch blocks."""
        past = merge_kv_blocks(tuple(branch_blocks))
        prefix_len = 0 if past is None else past[0][0].shape[2]
        prefix_mask = attention_mask.new_ones((input_ids.shape[0], prefix_len))
        attention = torch.cat([prefix_mask, attention_mask], dim=1)
        positions = _build_position_ids(prefix_mask, input_ids.shape[1])
        if torch.is_grad_enabled():
            embeddings = self.model.get_input_embeddings()(input_ids)
            flat_past = () if past is None else tuple(
                tensor for key, value in past for tensor in (key, value)
            )
            layer_count = len(flat_past) // 2

            def forward(values: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor, *flat: torch.Tensor):
                cache = None if not flat else tuple(
                    (flat[2 * index], flat[2 * index + 1]) for index in range(layer_count)
                )
                return self.model(
                    inputs_embeds=values, attention_mask=mask, position_ids=pos,
                    past_key_values=cache, use_cache=False, return_dict=True,
                ).logits

            logits = checkpoint(forward, embeddings, attention, positions, *flat_past, use_reentrant=False)
        else:
            logits = self.model(
                input_ids=input_ids, attention_mask=attention, position_ids=positions,
                past_key_values=past, use_cache=False, return_dict=True,
            ).logits
        last = attention_mask.sum(dim=1).long() - 1
        return logits[torch.arange(input_ids.shape[0], device=input_ids.device), last]


LatentKVProtocol = DistributedLatentProtocol

__all__ = [
    "KVBlock", "DistributedLatentProtocol", "LatentKVProtocol",
    "merge_kv_blocks", "slice_kv_block", "kv_wire_bytes",
]
