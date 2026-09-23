"""Differentiable fixed-route latent KV communication.

The protocol transports only a newly encoded ``(m + 2)`` position KV block:
``BEGIN``, ``m`` distilled positions, and ``END``.  KV blocks are tuples of
``(key, value)`` tensors, one pair per decoder layer, shaped
``[batch, heads, positions, head_dim]``.  Blocks are concatenated in stable
child/branch order before a relay re-encodes them with its private prompt.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from ..hf_data import IGNORE_INDEX
from ..hf_medlatent_h import _build_position_ids, _iter_key_value_pairs, _to_dynamic_cache
from ..losses import diagnosis_cross_entropy
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


def detach_kv_block(block: KVBlock) -> KVBlock:
    """Detach child summaries at the training-window boundary."""
    return tuple((key.detach(), value.detach()) for key, value in block)


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


def select_kv_rows(block: KVBlock, rows: int | Sequence[int]) -> KVBlock:
    """Select batch rows from a block without detaching its route gradient."""
    indices = [rows] if isinstance(rows, int) else list(rows)
    if not indices:
        raise ValueError("rows must not be empty")
    return tuple((key[indices], value[indices]) for key, value in block)


def ring_two_hop_branches(source_id: int, num_agents: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``(leaf, relay)`` paths in deterministic B2 ring branch order."""
    if num_agents < 5:
        raise ValueError("two-hop ring branches require at least five agents")
    if not 0 <= source_id < num_agents:
        raise ValueError("invalid source_id")
    relays = tuple(sorted(((source_id - 1) % num_agents, (source_id + 1) % num_agents)) )
    return tuple((((2 * relay - source_id) % num_agents), relay) for relay in relays)  # type: ignore[return-value]


def graph_two_hop_branches(graph: Any, source_id: int, *, max_branches: int = 2) -> tuple[tuple[int, int], ...]:
    """Return exactly two deterministic ``(leaf, relay)`` branches.

    M4 has one canonical route.  A graph that cannot provide two disjoint
    branches is invalid for the episode instead of silently degrading to a
    one-hop or single-branch route.
    """
    if max_branches != 2:
        raise ValueError("M4 routes require exactly two branches")
    from itertools import combinations

    source_neighbors = set(graph.neighbors(source_id))
    candidates: list[tuple[bool, int, int]] = []
    for relay in graph.neighbors(source_id):
        leaves = tuple(neighbor for neighbor in graph.neighbors(relay) if neighbor != source_id)
        candidates.extend((leaf in source_neighbors, relay, leaf) for leaf in leaves)

    ordered = sorted(candidates)
    for branch_count in (2,):
        valid = []
        for choice in combinations(ordered, branch_count):
            relays = {relay for _, relay, _ in choice}
            leaves = {leaf for _, _, leaf in choice}
            if len(relays) != branch_count or len(leaves) != branch_count or not relays.isdisjoint(leaves):
                continue
            routes = tuple(sorted((relay, leaf) for _, relay, leaf in choice))
            valid.append((sum(int(is_direct) for is_direct, _, _ in choice), routes))
        if valid:
            _, routes = min(valid)
            return tuple((leaf, relay) for relay, leaf in routes)
    raise ValueError(f"source {source_id} has no two node-disjoint two-hop branches")


def build_two_hop_route(
    graph: Any, source_id: int, *, block_shape: tuple[int, ...] | None = None,
    wire_bytes: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build the shared causal route record used by training and evaluation."""
    branches = graph_two_hop_branches(graph, source_id)
    route = []
    for branch, (leaf, relay) in enumerate(branches):
        if not graph.has_edge(source_id, relay) or not graph.has_edge(relay, leaf):
            raise ValueError("two-hop route contains a non-edge")
        if leaf == source_id or leaf == relay or relay == source_id:
            raise ValueError("two-hop route contains an invalid role assignment")
        route.extend((
            {"branch": branch, "sender": leaf, "receiver": relay, "round": 2, "kind": "PROPOSAL",
             "block_shape": block_shape, "wire_bytes": wire_bytes},
            {"branch": branch, "sender": relay, "receiver": source_id, "round": 3, "kind": "PROPOSAL",
             "block_shape": block_shape, "wire_bytes": wire_bytes},
        ))
    return tuple(route)


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
        if kwargs.get("past_key_values") is not None and self._requires_dynamic_cache(decoder):
            kwargs["past_key_values"] = _to_dynamic_cache(kwargs["past_key_values"])
        return decoder(**kwargs)

    @staticmethod
    def _requires_dynamic_cache(model: torch.nn.Module) -> bool:
        config = getattr(model, "config", None)
        return getattr(config, "model_type", None) == "qwen3"

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
                incoming: Sequence[KVBlock] = (), *, detach_incoming: bool = False) -> KVBlock:
        """Re-encode local private evidence and incoming child summaries."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both be [batch, sequence]")
        incoming_blocks = tuple(detach_kv_block(block) for block in incoming) if detach_incoming else tuple(incoming)
        incoming_block = merge_kv_blocks(incoming_blocks)
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

    def relay_rollout(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      child_blocks: Sequence[KVBlock] = (), *, detach_children: bool = True) -> KVBlock:
        """Encode local evidence together with ordered child summaries."""
        return self.rollout(input_ids, attention_mask, child_blocks, detach_incoming=detach_children)

    def source_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      branch_blocks: Sequence[KVBlock] = ()) -> torch.Tensor:
        """Return logits for a source prompt conditioned on final branch blocks."""
        logits = self._source_sequence_logits(input_ids, attention_mask, branch_blocks)
        last = attention_mask.sum(dim=1).long() - 1
        return logits[torch.arange(input_ids.shape[0], device=input_ids.device), last]

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 branch_blocks: Sequence[KVBlock] = (), *, max_new_tokens: int = 64,
                 eos_token_id: int | None = None, pad_token_id: int | None = None) -> torch.Tensor:
        """Greedy batched generation from a source prompt and returned blocks."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both be [batch, sequence]")
        past = merge_kv_blocks(tuple(branch_blocks))
        prefix_len = 0 if past is None else past[0][0].shape[2]
        prefix_mask = attention_mask.new_ones((input_ids.shape[0], prefix_len))
        attention = torch.cat([prefix_mask, attention_mask], dim=1)
        positions = _build_position_ids(prefix_mask, input_ids.shape[1])
        decoder_kwargs = {
            "input_ids": input_ids, "attention_mask": attention, "position_ids": positions,
            "past_key_values": past, "use_cache": True, "return_dict": True,
        }
        if past is not None and self._requires_dynamic_cache(self.model):
            decoder_kwargs["past_key_values"] = _to_dynamic_cache(past)
        outputs = self.model(**decoder_kwargs)
        cache = _as_block(outputs.past_key_values)
        source_lengths = attention_mask.sum(dim=1).long()
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        # CausalLM logits contain only the newly supplied source tokens;
        # cached branch positions are not part of the returned sequence.
        logits = outputs.logits[rows, source_lengths - 1, :]
        generated: list[torch.Tensor] = []
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            next_ids = logits.argmax(dim=-1)
            if pad_token_id is not None:
                next_ids = torch.where(finished, torch.full_like(next_ids, pad_token_id), next_ids)
            generated.append(next_ids)
            if eos_token_id is not None:
                finished = finished | (next_ids == eos_token_id)
                if bool(finished.all()):
                    break
            attention = torch.cat([attention, attention.new_ones((input_ids.shape[0], 1))], dim=1)
            outputs = self.model(
                input_ids=next_ids.unsqueeze(1), attention_mask=attention,
                position_ids=_build_position_ids(attention[:, :-1], 1),
                past_key_values=_to_dynamic_cache(cache) if self._requires_dynamic_cache(self.model) else cache,
                use_cache=True, return_dict=True,
            )
            cache = _as_block(outputs.past_key_values)
            logits = outputs.logits[:, -1, :]
        return torch.stack(generated, dim=1) if generated else input_ids.new_empty((input_ids.shape[0], 0))

    def source_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    target_ids: torch.Tensor, target_labels: torch.Tensor,
                    branch_blocks: Sequence[KVBlock] = ()) -> torch.Tensor:
        """Teacher-force a diagnosis after the source prompt and branch blocks."""
        if target_ids.ndim != 2 or target_labels.shape != target_ids.shape:
            raise ValueError("target_ids and target_labels must both be [batch, sequence]")
        if input_ids.shape[0] != target_ids.shape[0]:
            raise ValueError("source and target batches must have the same size")
        target_mask = (target_labels != IGNORE_INDEX).long()
        source_lengths = attention_mask.sum(dim=1).long()
        if torch.any(source_lengths <= 0):
            raise ValueError("each source row must contain at least one token")

        # Move valid target tokens directly after each row's valid source tokens.
        # This keeps one batched forward without placing the answer after source PADs.
        raw_ids = torch.cat([input_ids, target_ids], dim=1)
        raw_mask = torch.cat([attention_mask, target_mask], dim=1)
        order = torch.argsort((raw_mask == 0).to(torch.int8), dim=1, stable=True)
        full_ids = raw_ids.gather(1, order)
        full_mask = raw_mask.gather(1, order)
        logits = self._source_sequence_logits(full_ids, full_mask, branch_blocks)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        first = logits[rows, source_lengths - 1, :]
        target_positions = source_lengths.unsqueeze(1) + torch.arange(
            target_ids.shape[1], device=input_ids.device,
        ).unsqueeze(0)
        target_logits = logits.gather(
            1, target_positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
        )
        return diagnosis_cross_entropy(first, target_logits, target_labels, ignore_index=IGNORE_INDEX)

    def _source_sequence_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                                branch_blocks: Sequence[KVBlock]) -> torch.Tensor:
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
                decoder_kwargs = dict(
                    inputs_embeds=values, attention_mask=mask, position_ids=pos,
                    past_key_values=cache, use_cache=False, return_dict=True,
                )
                if cache is not None and self._requires_dynamic_cache(self.model):
                    decoder_kwargs["past_key_values"] = _to_dynamic_cache(cache)
                return self.model(**decoder_kwargs).logits

            logits = checkpoint(forward, embeddings, attention, positions, *flat_past, use_reentrant=False)
        else:
            decoder_kwargs = dict(
                input_ids=input_ids, attention_mask=attention, position_ids=positions,
                past_key_values=past, use_cache=False, return_dict=True,
            )
            if past is not None and self._requires_dynamic_cache(self.model):
                decoder_kwargs["past_key_values"] = _to_dynamic_cache(past)
            logits = self.model(**decoder_kwargs).logits
        return logits


class DecentralizedLatentNode:
    """One agent's trainable latent interface over a shared frozen backbone.

    The node owns its distiller, boundary embeddings, and optimizer.  Incoming
    child blocks are treated as transport inputs and detached at the relay
    boundary, so a node update never backpropagates into another node.
    """

    def __init__(self, node_id: int, model: torch.nn.Module, hidden_size: int,
                 *, num_latents: int = 8, device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None, learning_rate: float = 1e-4,
                 weight_decay: float = 0.01):
        if isinstance(node_id, bool) or node_id < 0:
            raise ValueError("node_id must be a non-negative integer")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.node_id = int(node_id)
        module_kwargs = {}
        if device is not None:
            module_kwargs["device"] = device
        if dtype is not None:
            module_kwargs["dtype"] = dtype
        self.distiller = LatentDistiller(hidden_size).to(**module_kwargs)
        self.boundary = BoundaryEmbeddings(hidden_size).to(**module_kwargs)
        self.protocol = DistributedLatentProtocol(
            model, self.distiller, self.boundary, num_latents=num_latents,
        )
        # Gossip synchronizes parameters only; zero-momentum SGD avoids stale
        # Adam moments after a peer average.
        self.optimizer = torch.optim.SGD(
            [*self.distiller.parameters(), *self.boundary.parameters()],
            lr=learning_rate, weight_decay=weight_decay,
        )

    @property
    def parameters(self):
        return tuple(self.distiller.parameters()) + tuple(self.boundary.parameters())

    def loss(self, *, mode: str, local_ids: torch.Tensor, local_mask: torch.Tensor,
             source_ids: torch.Tensor, source_mask: torch.Tensor,
             target_ids: torch.Tensor, target_labels: torch.Tensor,
             incoming: Sequence[KVBlock] = ()) -> torch.Tensor:
        """Compute one node-local diagnosis loss in explicit local/relay mode."""
        if mode == "local":
            if incoming:
                raise ValueError("local mode cannot receive incoming blocks")
            block = self.protocol.rollout(local_ids, local_mask, ())
        elif mode == "relay":
            block = self.protocol.relay_rollout(
                local_ids, local_mask, incoming, detach_children=True,
            )
        else:
            raise ValueError("mode must be 'local' or 'relay'")
        return self.protocol.source_loss(
            source_ids, source_mask, target_ids, target_labels, (block,)
        )

    def role_loss(self, *, role: str, local_ids: torch.Tensor, local_mask: torch.Tensor,
                  public_ids: torch.Tensor, public_mask: torch.Tensor,
                  source_ids: torch.Tensor, source_mask: torch.Tensor,
                  target_ids: torch.Tensor, target_labels: torch.Tensor,
                  incoming: Sequence[KVBlock] = ()) -> tuple[torch.Tensor, KVBlock | tuple[()]]:
        """Compute one explicit M4 role loss and return its fresh block.

        Leaf and relay supervision uses only the public query.  The source
        supervision uses its private prompt and only returned relay blocks.
        Incoming blocks are transport payloads and are detached at every
        boundary.
        """
        if role == "leaf":
            if incoming:
                raise ValueError("leaf role cannot receive incoming blocks")
            block = self.protocol.rollout(local_ids, local_mask)
            loss = self.protocol.source_loss(public_ids, public_mask, target_ids, target_labels, (block,))
        elif role == "relay":
            if len(incoming) != 1:
                raise ValueError("relay role requires one leaf block")
            block = self.protocol.relay_rollout(local_ids, local_mask, incoming, detach_children=True)
            loss = self.protocol.source_loss(public_ids, public_mask, target_ids, target_labels, (block,))
        elif role == "source":
            if len(incoming) != 2:
                raise ValueError("source role requires two relay blocks")
            # The source replica also participates in the interface: it
            # consumes the two returned relay blocks with its private prompt,
            # then predicts from that source-local representation.
            block = self.protocol.rollout(local_ids, local_mask, incoming, detach_incoming=True)
            loss = self.protocol.source_loss(source_ids, source_mask, target_ids, target_labels, (block,))
        else:
            raise ValueError("role must be 'leaf', 'relay', or 'source'")
        return loss, block

    def train_step(self, *, mode: str, local_ids: torch.Tensor,
                   local_mask: torch.Tensor, source_ids: torch.Tensor,
                   source_mask: torch.Tensor, target_ids: torch.Tensor,
                   target_labels: torch.Tensor, incoming: Sequence[KVBlock] = ()) -> float:
        """Run one isolated optimizer update and return its detached loss."""
        self.distiller.train()
        self.boundary.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(
            mode=mode, local_ids=local_ids, local_mask=local_mask,
            source_ids=source_ids, source_mask=source_mask,
            target_ids=target_ids, target_labels=target_labels, incoming=incoming,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def train_role_step(self, *, role: str, incoming: Sequence[KVBlock] = (),
                        **tensors: torch.Tensor) -> tuple[float, KVBlock | tuple[()]]:
        """Apply one optimizer step for one M4 role."""
        self.distiller.train()
        self.boundary.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, block = self.role_loss(role=role, incoming=incoming, **tensors)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
        self.optimizer.step()
        return float(loss.detach().cpu()), block

    def state(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "distiller": self.distiller.state(),
            "boundary": self.boundary.state(),
        }


def graph_broadcast_tree(graph: Any, source_id: int, *, max_fanout: int = 2) -> tuple[tuple[int, ...], dict[int, int | None]]:
    """Return a deterministic BFS tree with bounded per-node fan-out."""
    if source_id not in graph.agent_ids:
        raise ValueError(f"invalid source_id: {source_id}")
    if max_fanout <= 0:
        raise ValueError("max_fanout must be positive")
    parent: dict[int, int | None] = {source_id: None}
    order: list[int] = []
    queue = deque([source_id])
    while queue:
        current = queue.popleft()
        order.append(current)
        unvisited = [neighbor for neighbor in graph.neighbors(current) if neighbor not in parent]
        for neighbor in unvisited[:max_fanout]:
            parent[neighbor] = current
            queue.append(neighbor)
    if len(parent) != graph.num_agents:
        raise ValueError("training route requires a connected graph")
    return tuple(order), parent


class NodeLocalLatentTrainer:
    """Own one private store and build one node-local training example."""

    def __init__(self, node: DecentralizedLatentNode, store: Any, *, tokenizer: Any,
                 hospital_id: int, max_prompt_length: int = 320,
                 max_target_length: int = 64, device: torch.device | str = "cpu"):
        self.node = node
        self.store = store
        self.tokenizer = tokenizer
        self.hospital_id = int(hospital_id)
        self.max_prompt_length = int(max_prompt_length)
        self.max_target_length = int(max_target_length)
        self.device = torch.device(device)

    def train_episode(self, episode: Any, *, mode: str,
                      incoming: Sequence[KVBlock] = ()) -> tuple[float, KVBlock]:
        return self.train_batch((episode,), mode=mode, incoming=incoming)

    def prepare_batch(self, episodes: Sequence[Any]) -> dict[str, torch.Tensor]:
        """Materialize only this node's private prompt and public query."""
        from ..hf_data import _pad
        from .medical import build_agent_prompt_batch

        episodes = tuple(episodes)
        if not episodes:
            raise ValueError("episodes must not be empty")
        rows = build_agent_prompt_batch(
            episodes, self.store, hospital_id=self.hospital_id, tokenizer=self.tokenizer,
            max_prompt_length=self.max_prompt_length, max_target_length=self.max_target_length,
        )
        local_rows = [row["local_ids"] for row in rows]
        public_rows = [row["public_query_ids"] for row in rows]
        source_rows = [row["local_ids"] + row["host_question_ids"] for row in rows]
        local_ids, local_mask = _pad(local_rows, self.tokenizer.pad_token_id)
        public_ids, public_mask = _pad(public_rows, self.tokenizer.pad_token_id)
        source_ids, source_mask = _pad(source_rows, self.tokenizer.pad_token_id)
        target_ids, target_mask = _pad([row["target_ids"] for row in rows], self.tokenizer.pad_token_id)
        tensors = {
            "local_ids": local_ids, "local_mask": local_mask,
            "public_ids": public_ids, "public_mask": public_mask,
            "source_ids": source_ids, "source_mask": source_mask,
            "target_ids": target_ids,
            "target_labels": target_ids.masked_fill(target_mask == 0, IGNORE_INDEX),
        }
        return {key: value.to(self.device) for key, value in tensors.items()}

    def train_batch(self, episodes: Sequence[Any], *, mode: str,
                    incoming: Sequence[KVBlock] = ()) -> tuple[float, KVBlock]:
        """Train one node on a batch of same-route episodes."""
        episodes = tuple(episodes)
        tensors = self.prepare_batch(episodes)
        loss = self.node.train_step(
            mode=mode, incoming=incoming,
            local_ids=tensors["local_ids"], local_mask=tensors["local_mask"],
            source_ids=tensors["source_ids"], source_mask=tensors["source_mask"],
            target_ids=tensors["target_ids"], target_labels=tensors["target_labels"],
        )
        with torch.no_grad():
            local_ids, local_mask = tensors["local_ids"], tensors["local_mask"]
            if mode == "local":
                outgoing = self.node.protocol.rollout(local_ids, local_mask)
            else:
                outgoing = self.node.protocol.relay_rollout(
                    local_ids, local_mask, incoming, detach_children=True,
                )
        return loss, outgoing


def canonical_two_hop_losses(
    *, nodes: Mapping[int, DecentralizedLatentNode], trainers: Mapping[int, NodeLocalLatentTrainer],
    graph: Any, episode: Any, lambda_leaf: float = 0.5, lambda_relay: float = 0.5,
    lambda_src: float = 1.0,
) -> dict[str, Any]:
    """Build one canonical M4 computation graph for a single episode.

    Every prompt is materialized by its owning trainer.  Returned blocks are
    detached before crossing an edge, and the source receives relay blocks
    only.  Callers can backpropagate ``total_loss`` or inspect the role losses.
    """
    if set(nodes) != set(graph.agent_ids) or set(trainers) != set(graph.agent_ids):
        raise ValueError("nodes and trainers must contain exactly the graph agents")
    for value, name in ((lambda_leaf, "lambda_leaf"), (lambda_relay, "lambda_relay"), (lambda_src, "lambda_src")):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    branches = graph_two_hop_branches(graph, episode.source_id)
    source_tensors = trainers[episode.source_id].prepare_batch((episode,))
    leaf_blocks: list[KVBlock] = []
    relay_blocks: list[KVBlock] = []
    leaf_losses: list[torch.Tensor] = []
    relay_losses: list[torch.Tensor] = []
    for leaf_id, relay_id in branches:
        leaf_tensors = trainers[leaf_id].prepare_batch((episode,))
        relay_tensors = trainers[relay_id].prepare_batch((episode,))
        leaf_loss, leaf_block = nodes[leaf_id].role_loss(role="leaf", **leaf_tensors)
        relay_loss, relay_block = nodes[relay_id].role_loss(
            role="relay", incoming=(detach_kv_block(leaf_block),), **relay_tensors,
        )
        leaf_losses.append(leaf_loss)
        relay_losses.append(relay_loss)
        leaf_blocks.append(detach_kv_block(leaf_block))
        relay_blocks.append(detach_kv_block(relay_block))
    source_loss, _ = nodes[episode.source_id].role_loss(
        role="source", incoming=tuple(relay_blocks), **source_tensors,
    )
    leaf_loss = torch.stack(leaf_losses).mean()
    relay_loss = torch.stack(relay_losses).mean()
    total = lambda_leaf * leaf_loss + lambda_relay * relay_loss + lambda_src * source_loss
    shape = tuple(int(value) for value in relay_blocks[0][0][0].shape) if relay_blocks else None
    return {
        "route": build_two_hop_route(
            graph, episode.source_id, block_shape=shape,
            wire_bytes=kv_wire_bytes(relay_blocks[0]) if relay_blocks else 0,
        ),
        "leaf_blocks": tuple(leaf_blocks), "relay_blocks": tuple(relay_blocks),
        "leaf_loss": leaf_loss, "relay_loss": relay_loss, "source_loss": source_loss,
        "total_loss": total,
    }


def metropolis_mixing_weights(graph: Any) -> dict[int, dict[int, float]]:
    """Return deterministic symmetric, row-stochastic graph mixing weights."""
    if not hasattr(graph, "agent_ids") or not hasattr(graph, "neighbors"):
        raise TypeError("graph must provide agent_ids and neighbors")
    degrees = {agent_id: len(graph.neighbors(agent_id)) for agent_id in graph.agent_ids}
    weights: dict[int, dict[int, float]] = {}
    for agent_id in graph.agent_ids:
        row = {
            neighbor: 1.0 / (1.0 + max(degrees[agent_id], degrees[neighbor]))
            for neighbor in graph.neighbors(agent_id)
        }
        row[agent_id] = 1.0 - sum(row.values())
        if row[agent_id] < -1e-12:
            raise ValueError("graph mixing row has negative self weight")
        row[agent_id] = max(0.0, row[agent_id])
        weights[agent_id] = dict(sorted(row.items()))
    return weights


def _node_parameter_snapshots(nodes: Mapping[int, DecentralizedLatentNode], graph: Any) -> dict[int, dict[str, torch.Tensor]]:
    expected = tuple(graph.agent_ids)
    if tuple(sorted(nodes)) != expected:
        raise ValueError("nodes must contain exactly the graph agent IDs")
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    for agent_id in expected:
        node = nodes[agent_id]
        current = {name: parameter.detach().clone() for name, parameter in node.distiller.named_parameters()}
        current.update({f"boundary.{name}": parameter.detach().clone()
                        for name, parameter in node.boundary.named_parameters()})
        shapes = {name: tuple(value.shape) for name, value in current.items()}
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise ValueError("all node interface parameters must have identical shapes")
        snapshots[agent_id] = current
    return snapshots


def gossip_node_interfaces(
    nodes: Mapping[int, DecentralizedLatentNode], graph: Any,
    *, mixing_weights: Mapping[int, Mapping[int, float]] | None = None,
) -> dict[int, dict[int, float]]:
    """Synchronize node-local trainable interfaces with one atomic gossip step."""
    snapshots = _node_parameter_snapshots(nodes, graph)
    weights = metropolis_mixing_weights(graph) if mixing_weights is None else {
        int(agent_id): {int(peer): float(weight) for peer, weight in row.items()}
        for agent_id, row in mixing_weights.items()
    }
    expected = tuple(graph.agent_ids)
    if tuple(sorted(weights)) != expected:
        raise ValueError("mixing weights must contain exactly the graph agent IDs")
    for agent_id in expected:
        row = weights[agent_id]
        if set(row) != set(graph.neighbors(agent_id)) | {agent_id}:
            raise ValueError("mixing weights must match graph neighbors")
        if any(weight < 0 for weight in row.values()) or abs(sum(row.values()) - 1.0) > 1e-6:
            raise ValueError("each mixing row must be non-negative and sum to one")
        for peer_id, weight in row.items():
            if peer_id != agent_id and abs(weight - weights[peer_id].get(agent_id, -1.0)) > 1e-6:
                raise ValueError("mixing weights must be symmetric")
    for agent_id in expected:
        node = nodes[agent_id]
        with torch.no_grad():
            for name, parameter in node.distiller.named_parameters():
                value = sum(weights[agent_id][peer] * snapshots[peer][name]
                            for peer in weights[agent_id])
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            for name, parameter in node.boundary.named_parameters():
                key = f"boundary.{name}"
                value = sum(weights[agent_id][peer] * snapshots[peer][key]
                            for peer in weights[agent_id])
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return {agent_id: dict(row) for agent_id, row in weights.items()}


def interface_consensus_distance(nodes: Mapping[int, DecentralizedLatentNode], graph: Any) -> float:
    """Return RMS distance of node interfaces from their parameter centroid."""
    snapshots = _node_parameter_snapshots(nodes, graph)
    vectors = [torch.cat([value.float().reshape(-1) for value in snapshots[agent_id].values()])
               for agent_id in graph.agent_ids]
    centroid = torch.stack(vectors).mean(dim=0)
    return float(torch.stack([(vector - centroid).pow(2).mean() for vector in vectors]).mean().sqrt())

# NOTE: Unused - Legacy
# def two_hop_ring_blocks(protocol: DistributedLatentProtocol, *, leaf_ids: torch.Tensor,
#                         leaf_mask: torch.Tensor, relay_ids: torch.Tensor,
#                         relay_mask: torch.Tensor) -> tuple[KVBlock, KVBlock]:
#     """Return the two final branch summaries from batched leaf/relay passes."""
#     if leaf_ids.shape[0] != 2 or relay_ids.shape[0] != 2:
#         raise ValueError("leaf and relay batches must contain the two ring branches")
#     leaf_blocks = protocol.rollout(leaf_ids, leaf_mask)
#     relay_blocks = protocol.relay_rollout(relay_ids, relay_mask, [leaf_blocks], detach_children=False)
#     return select_kv_rows(relay_blocks, 0), select_kv_rows(relay_blocks, 1)

# def run_two_hop_ring_query(protocol: DistributedLatentProtocol, *, source_ids: torch.Tensor,
#                            source_mask: torch.Tensor, leaf_ids: torch.Tensor, leaf_mask: torch.Tensor,
#                            relay_ids: torch.Tensor, relay_mask: torch.Tensor) -> torch.Tensor:
#     """Run one query with two hospitals batched at each fixed ring hop."""
#     branches = two_hop_ring_blocks(protocol, leaf_ids=leaf_ids, leaf_mask=leaf_mask,
#                                    relay_ids=relay_ids, relay_mask=relay_mask)
#     return protocol.source_logits(source_ids, source_mask, branches)


def batched_rollout(protocol: DistributedLatentProtocol, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, incoming: Sequence[KVBlock] = (), *,
                    detach_children: bool = True) -> KVBlock:
    """Run one shared local/relay operator over a padded batch."""
    return protocol.rollout(input_ids, attention_mask, incoming, detach_incoming=detach_children)


def batched_relay_aggregate(protocol: DistributedLatentProtocol, input_ids: torch.Tensor,
                            attention_mask: torch.Tensor, child_blocks: Sequence[KVBlock] = (),
                            *, detach_children: bool = True) -> KVBlock:
    """Explicit relay entry point used by the batched pilot runner."""
    return protocol.relay_rollout(input_ids, attention_mask, child_blocks, detach_children=detach_children)


def batched_pilot_loss(
    protocol: DistributedLatentProtocol, *, source_ids: torch.Tensor, source_mask: torch.Tensor,
    target_ids: torch.Tensor, target_labels: torch.Tensor, leaf_ids: torch.Tensor,
    leaf_mask: torch.Tensor, relay_ids: torch.Tensor, relay_mask: torch.Tensor,
    use_relay: bool,
) -> torch.Tensor:
    """Compute one local or detached-relay pilot loss for ``2 * batch`` branches."""
    batch_size = source_ids.shape[0]
    if leaf_ids.shape[0] != batch_size * 2 or relay_ids.shape[0] != batch_size * 2:
        raise ValueError("leaf and relay rows must contain two branches per source row")
    if use_relay:
        with torch.no_grad():
            leaf_blocks = protocol.rollout(leaf_ids, leaf_mask)
        final_blocks = protocol.relay_rollout(relay_ids, relay_mask, [leaf_blocks], detach_children=True)
    else:
        final_blocks = protocol.rollout(leaf_ids, leaf_mask)
    branches = tuple(
        select_kv_rows(final_blocks, list(range(branch, batch_size * 2, 2)))
        for branch in (0, 1)
    )
    return protocol.source_loss(source_ids, source_mask, target_ids, target_labels, branches)


def tiny_overfit(
    protocol: DistributedLatentProtocol, optimizer: torch.optim.Optimizer, *,
    source_ids: torch.Tensor, source_mask: torch.Tensor, target_ids: torch.Tensor,
    target_labels: torch.Tensor, leaf_ids: torch.Tensor, leaf_mask: torch.Tensor,
    relay_ids: torch.Tensor, relay_mask: torch.Tensor, steps: int,
) -> dict[str, float]:
    """Overfit one padded batch, alternating local and detached-relay losses."""
    if steps <= 0:
        return {"steps": 0.0}
    losses: dict[str, list[float]] = {"local": [], "relay": []}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = batched_pilot_loss(
            protocol, source_ids=source_ids, source_mask=source_mask,
            target_ids=target_ids, target_labels=target_labels,
            leaf_ids=leaf_ids, leaf_mask=leaf_mask, relay_ids=relay_ids, relay_mask=relay_mask,
            use_relay=bool(step % 2),
        )
        mode = "relay" if step % 2 else "local"
        losses[mode].append(float(loss.detach().cpu()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(protocol.distiller.parameters()) + list(protocol.boundary.parameters()), 1.0)
        optimizer.step()
    return {
        "steps": float(steps),
        **{f"{mode}_{point}_loss": values[0 if point == "initial" else -1]
           for mode, values in losses.items() if values for point in ("initial", "final")},
    }


def _pilot_batch_tensors(*, episodes: Sequence[Any], rows_by_case: Mapping[str, Mapping[str, Any]],
                         graph: Any, pad_token_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Assemble one padded source/leaf/relay batch in stable branch order."""
    from ..hf_data import _pad

    routes = [tuple((event["sender"], event["receiver"])
                    for event in build_two_hop_route(graph, episode.source_id)
                    if event["round"] == 2) for episode in episodes]
    if any(len(route) != 2 for route in routes):
        raise ValueError("pilot graph must provide two two-hop branches per source")
    leaf_rows = [rows_by_case[e.query.case_id]["hospital_ids_all"][leaf]
                 for e, route in zip(episodes, routes) for leaf, _ in route]
    relay_rows = [rows_by_case[e.query.case_id]["hospital_ids_all"][relay]
                  for e, route in zip(episodes, routes) for _, relay in route]
    source_rows = [rows_by_case[e.query.case_id]["hospital_ids_all"][e.source_id]
                   + rows_by_case[e.query.case_id]["host_question_ids"] for e in episodes]
    target_rows = [rows_by_case[e.query.case_id]["target_ids"] for e in episodes]
    leaf_ids, leaf_mask = _pad(leaf_rows, pad_token_id)
    relay_ids, relay_mask = _pad(relay_rows, pad_token_id)
    source_ids, source_mask = _pad(source_rows, pad_token_id)
    target_ids, target_mask = _pad(target_rows, pad_token_id)
    return {
        "source_ids": source_ids.to(device), "source_mask": source_mask.to(device),
        "target_ids": target_ids.to(device),
        "target_labels": target_ids.masked_fill(target_mask == 0, IGNORE_INDEX).to(device),
        "leaf_ids": leaf_ids.to(device), "leaf_mask": leaf_mask.to(device),
        "relay_ids": relay_ids.to(device), "relay_mask": relay_mask.to(device),
    }


def accumulation_steps_for(*, batch_size: int, effective_batch_size: int) -> int:
    """Return MedLatent-H-style micro-batches per optimizer update."""
    if batch_size <= 0 or effective_batch_size <= 0:
        raise ValueError("batch_size and effective_batch_size must be positive")
    if effective_batch_size < batch_size or effective_batch_size % batch_size:
        raise ValueError("effective_batch_size must be divisible by and at least batch_size")
    return effective_batch_size // batch_size


def save_distributed_latent_checkpoint(path: str | Path, *, protocol: DistributedLatentProtocol,
                                       model_name: str, route: Mapping[str, object], optimizer: torch.optim.Optimizer,
                                       training: Mapping[str, object]) -> None:
    """Save the only trainable modules plus fixed protocol metadata."""
    payload = {
        "module_type": "DistributedLatentKV",
        "model_name": model_name,
        "num_latents": protocol.num_latents,
        "route": dict(route),
        "distiller": protocol.distiller.state(),
        "boundary": protocol.boundary.state(),
        "optimizer": optimizer.state_dict(),
        "training": dict(training),
    }
    torch.save(payload, path)


def save_decentralized_latent_checkpoint(
    path: str | Path, *, nodes: Mapping[int, DecentralizedLatentNode], model_name: str,
    route: Mapping[str, object], training: Mapping[str, object],
    gossip: Mapping[str, object], optimizer: bool = True,
) -> None:
    """Save per-node trainable interfaces and decentralized training metadata."""
    if not nodes:
        raise ValueError("nodes must not be empty")
    payload = {
        "module_type": "DecentralizedLatentKV",
        "model_name": model_name,
        "num_latents": next(iter(nodes.values())).protocol.num_latents if nodes else 0,
        "route": dict(route),
        "nodes": {
            int(node_id): {
                **node.state(),
                **({"optimizer": node.optimizer.state_dict()} if optimizer else {}),
            }
            for node_id, node in nodes.items()
        },
        "gossip": dict(gossip),
        "training": dict(training),
    }
    torch.save(payload, path)


def load_distributed_latent_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if payload.get("module_type") not in {"DistributedLatentKV", "DecentralizedLatentKV"}:
        raise ValueError(f"Expected distributed latent checkpoint, got {payload.get('module_type')!r}")
    return payload


load_decentralized_latent_checkpoint = load_distributed_latent_checkpoint


def train_distributed_latent_real(*, model_name: str, train_file: str | Path | None = None,
                                  query_file: str | Path | None = None,
                                  hospital_dir: str | Path, output_dir: str | Path,
                                  hpo_embeddings_file: str | Path, hpo_ic_file: str | Path,
                                  validation_file: str | Path | None = None,
                                  num_agents: int = 5, pilot_hospital_ids: Sequence[int] | None = None,
                                  num_latents: int = 8, batch_size: int = 1,
                                  effective_batch_size: int = 8, graph_seed: int = 42,
                                  tiny_overfit_steps: int = 16,
                                  max_prompt_length: int = 320, max_target_length: int = 64,
                                  epochs: int = 1, max_steps: int = 0, learning_rate: float = 1e-4,
                                  weight_decay: float = 0.01, seed: int = 42,
                                  device: str = "cuda", dtype: str = "bfloat16",
                                  local_files_only: bool = False,
                                  training_mode: str = "centralized", local_steps: int = 0,
                                  sync_interval: int = 32) -> dict[str, float]:
    """Train the centralized M3 operator or decentralized local replicas."""
    query_file = query_file or train_file
    if query_file is None:
        raise ValueError("query_file is required")
    if training_mode not in {"centralized", "decentralized"}:
        raise ValueError("training_mode must be 'centralized' or 'decentralized'")
    if local_steps < 0:
        raise ValueError("local_steps must be non-negative")
    if sync_interval <= 0:
        raise ValueError("sync_interval must be positive")
    if num_agents != 5:
        raise ValueError("the M3 pilot currently requires num_agents=5")
    if epochs <= 0 or max_steps < 0 or tiny_overfit_steps < 0:
        raise ValueError("epochs must be positive and max_steps non-negative")
    accumulation_steps = accumulation_steps_for(
        batch_size=batch_size, effective_batch_size=effective_batch_size,
    )
    torch.manual_seed(seed)
    resolved_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..hf_data import MedLatentDiagnosisDataset
    from .medical import load_medical_split, sample_balanced_sources, select_pilot_hospital_ids
    from .graph import shortcut_ring_graph

    print(f"latent train: loading model={model_name} device={resolved_device}", flush=True)

    pilot_ids = select_pilot_hospital_ids(hospital_dir, num_agents=num_agents,
                                          pilot_hospital_ids=pilot_hospital_ids)
    graph = shortcut_ring_graph(num_agents, seed=graph_seed, num_shortcuts=1)
    split_paths = {"train": str(query_file)}
    try:
        from .medical import q4r6_paths
        qpaths = q4r6_paths(Path(query_file).parent)
        split_paths.update({"val": str(qpaths.validation), "test": str(qpaths.test)})
    except (FileNotFoundError, ValueError):
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    ).to(resolved_device)
    episodes = sample_balanced_sources(
        load_medical_split(query_file), split=Path(query_file).stem,
        num_agents=num_agents, seed=seed,
    )
    if training_mode == "decentralized":
        return _train_decentralized_nodes(
            model=model, model_name=model_name, tokenizer=tokenizer,
            episodes=episodes, output_dir=output_dir, graph=graph,
            pilot_ids=pilot_ids, split_paths=split_paths, hospital_dir=hospital_dir,
            hpo_embeddings_file=hpo_embeddings_file, hpo_ic_file=hpo_ic_file,
            num_latents=num_latents, learning_rate=learning_rate,
            weight_decay=weight_decay, local_steps=local_steps, epochs=epochs,
            batch_size=batch_size, seed=seed,
            sync_interval=sync_interval,
            max_prompt_length=max_prompt_length, max_target_length=max_target_length,
            device=resolved_device,
        )
    protocol = DistributedLatentProtocol(
        model,
        LatentDistiller(int(model.config.hidden_size)).to(device=resolved_device, dtype=torch_dtype),
        BoundaryEmbeddings(int(model.config.hidden_size)).to(device=resolved_device, dtype=torch_dtype),
        num_latents=num_latents,
    )
    dataset = MedLatentDiagnosisDataset(
        data_file=query_file, hospital_dir=hospital_dir, tokenizer=tokenizer,
        num_hospitals=num_agents, max_prompt_length=max_prompt_length,
        max_target_length=max_target_length, hpo_embeddings_file=hpo_embeddings_file,
        hpo_ic_file=hpo_ic_file, hospital_ids=list(pilot_ids),
    )
    rows_by_case = {row["case_id"]: row for row in dataset}
    updates_per_epoch = math.ceil(len(episodes) / effective_batch_size)
    total_updates = updates_per_epoch * epochs
    if max_steps:
        total_updates = min(total_updates, max_steps)
    log_interval = max(1, total_updates // 20)
    print(
        f"latent train: episodes={len(episodes)} batch_size={batch_size} "
        f"effective_batch_size={effective_batch_size} accumulation_steps={accumulation_steps} "
        f"target_updates={total_updates}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        list(protocol.distiller.parameters()) + list(protocol.boundary.parameters()),
        lr=learning_rate, weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    updates = 0
    micro_batches = 0
    last_loss = float("nan")
    protocol.distiller.train()
    protocol.boundary.train()
    tiny_batch = _pilot_batch_tensors(
        episodes=episodes[:min(batch_size, len(episodes))], rows_by_case=rows_by_case,
        graph=graph, pad_token_id=tokenizer.pad_token_id, device=resolved_device,
    )
    initial_distiller = {name: value.detach().clone() for name, value in protocol.distiller.state_dict().items()}
    initial_boundary = {name: value.detach().clone() for name, value in protocol.boundary.state_dict().items()}
    if tiny_overfit_steps:
        print(f"latent train: tiny overfit steps={tiny_overfit_steps}", flush=True)
    tiny = tiny_overfit(protocol, optimizer, steps=tiny_overfit_steps, **tiny_batch)
    if tiny_overfit_steps:
        print(
            "latent train: tiny overfit "
            f"local={tiny['local_initial_loss']:.4f}->{tiny['local_final_loss']:.4f} "
            f"relay={tiny['relay_initial_loss']:.4f}->{tiny['relay_final_loss']:.4f}",
            flush=True,
        )
    protocol.distiller.load_state_dict(initial_distiller)
    protocol.boundary.load_state_dict(initial_boundary)
    optimizer = torch.optim.AdamW(
        list(protocol.distiller.parameters()) + list(protocol.boundary.parameters()),
        lr=learning_rate, weight_decay=weight_decay,
    )
    start_time = time.perf_counter()
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    for epoch in range(epochs):
        order = torch.randperm(len(episodes), generator=generator).tolist()
        optimizer.zero_grad(set_to_none=True)
        pending_micro_batches = 0
        for start in range(0, len(order), batch_size):
            batch = [episodes[position] for position in order[start:start + batch_size]]
            tensors = _pilot_batch_tensors(episodes=batch, rows_by_case=rows_by_case, graph=graph,
                                           pad_token_id=tokenizer.pad_token_id, device=resolved_device)
            loss = batched_pilot_loss(protocol, use_relay=bool(micro_batches % 2), **tensors)
            (loss / accumulation_steps).backward()
            micro_batches += 1
            pending_micro_batches += 1
            last_loss = float(loss.detach().cpu())
            is_last = start + batch_size >= len(order)
            if pending_micro_batches == accumulation_steps or is_last:
                if pending_micro_batches < accumulation_steps:
                    scale = accumulation_steps / pending_micro_batches
                    for parameter in list(protocol.distiller.parameters()) + list(protocol.boundary.parameters()):
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(list(protocol.distiller.parameters()) + list(protocol.boundary.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                pending_micro_batches = 0
                if updates == 1 or updates % log_interval == 0 or updates == total_updates:
                    print(
                        f"latent train: update={updates}/{total_updates} "
                        f"loss={last_loss:.4f}",
                        flush=True,
                    )
            if max_steps and updates >= max_steps:
                break
        if max_steps and updates >= max_steps:
            break
    elapsed_seconds = time.perf_counter() - start_time
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(resolved_device)) if resolved_device.type == "cuda" else 0
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_distributed_latent_checkpoint(
        output / "distributed_latent_final.pt", protocol=protocol, model_name=model_name,
        route={"num_agents": num_agents, "rounds": 4, "max_fanout": 2, "R": 4, "k": 2, "m": num_latents,
               "graph_seed": graph_seed, "pilot_hospital_ids": list(pilot_ids),
               "graph_edges": [list(edge) for edge in graph.edge_list()], "split_paths": split_paths}, optimizer=optimizer,
        training={"seed": seed, "epochs": epoch + 1, "updates": updates, "micro_batches": micro_batches,
                  "effective_batch_size": effective_batch_size, "accumulation_steps": accumulation_steps, "last_loss": last_loss,
                  "tiny_overfit": tiny, "elapsed_seconds": elapsed_seconds, "peak_memory_bytes": peak_memory_bytes},
    )
    (output / "training_summary.json").write_text(json.dumps({
        "model_name": model_name, "num_agents": num_agents, "num_latents": num_latents,
        "query_file": str(query_file), "hospital_dir": str(hospital_dir),
        "pilot_hospital_ids": list(pilot_ids), "graph_edges": [list(edge) for edge in graph.edge_list()],
        "split_paths": split_paths, "R": 4, "k": 2,
        "updates": updates, "micro_batches": micro_batches, "effective_batch_size": effective_batch_size,
        "accumulation_steps": accumulation_steps, "last_loss": last_loss, "tiny_overfit": tiny,
        "elapsed_seconds": elapsed_seconds, "peak_memory_bytes": peak_memory_bytes,
    }, indent=2) + "\n", encoding="utf-8")
    return {"updates": float(updates), "last_loss": last_loss,
            "elapsed_seconds": elapsed_seconds, "peak_memory_bytes": float(peak_memory_bytes)}


def _train_decentralized_nodes(*, model: torch.nn.Module, model_name: str, tokenizer: Any,
                               episodes: Sequence[Any],
                               output_dir: str | Path, graph: Any, pilot_ids: Sequence[int],
                               split_paths: Mapping[str, str], hospital_dir: str | Path,
                               hpo_embeddings_file: str | Path, hpo_ic_file: str | Path,
                               num_latents: int,
                               learning_rate: float, weight_decay: float, local_steps: int,
                               epochs: int, batch_size: int, seed: int, sync_interval: int,
                               max_prompt_length: int, max_target_length: int,
                               device: torch.device) -> dict[str, float]:
    """Train node-local replicas on the canonical two-hop route."""
    if not episodes:
        raise ValueError("decentralized training requires at least one episode")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total_episodes = local_steps if local_steps > 0 else len(episodes) * epochs
    scheduled = [episodes[index % len(episodes)] for index in range(total_episodes)]
    random.Random(seed).shuffle(scheduled)
    batches = [scheduled[start:start + batch_size]
               for start in range(0, len(scheduled), batch_size)]
    total_steps = len(batches)
    hidden_size = int(model.config.hidden_size)
    dtype = model.get_input_embeddings().weight.dtype
    from .medical import load_hospital_private_stores
    reference = DecentralizedLatentNode(
        0, model, hidden_size, num_latents=num_latents,
        device=device, dtype=dtype, learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    nodes = {0: reference}
    for node_id in range(1, len(pilot_ids)):
        node = DecentralizedLatentNode(
            node_id, model, hidden_size, num_latents=num_latents,
            device=device, dtype=dtype, learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        node.distiller.load_state_dict(reference.distiller.state_dict())
        node.boundary.load_state_dict(reference.boundary.state_dict())
        nodes[node_id] = node
    trainers = {}
    for node_id in nodes:
        local_store = load_hospital_private_stores(
            hospital_dir, num_agents=1, hospital_ids=[pilot_ids[node_id]],
            hpo_embeddings_file=hpo_embeddings_file, hpo_ic_file=hpo_ic_file,
        )[0]
        trainers[node_id] = NodeLocalLatentTrainer(
            nodes[node_id], local_store, tokenizer=tokenizer,
            hospital_id=pilot_ids[node_id], max_prompt_length=max_prompt_length,
            max_target_length=max_target_length, device=device,
        )
    start_time = time.perf_counter()
    losses: list[float] = []
    gossip_rounds = 0
    mixing_weights = metropolis_mixing_weights(graph)
    log_interval = max(1, sync_interval)
    print(
        f"decentralized train: episodes={total_episodes} batches={total_steps} "
        f"batch_size={batch_size} nodes={len(nodes)} epochs={epochs} sync_interval={sync_interval}",
        flush=True,
    )
    for step, batch in enumerate(batches):
        step_losses: list[float] = []
        for episode in batch:
            for node in nodes.values():
                node.optimizer.zero_grad(set_to_none=True)
            result = canonical_two_hop_losses(
                nodes=nodes, trainers=trainers, graph=graph, episode=episode,
            )
            result["total_loss"].backward()
            for node in nodes.values():
                torch.nn.utils.clip_grad_norm_(node.parameters, 1.0)
                node.optimizer.step()
            value = float(result["total_loss"].detach().cpu())
            losses.append(value)
            step_losses.append(value)
        if (step + 1) % sync_interval == 0:
            mixing_weights = gossip_node_interfaces(nodes, graph, mixing_weights=mixing_weights)
            gossip_rounds += 1
        if step == 0 or (step + 1) % log_interval == 0 or step + 1 == total_steps:
            gossip_note = f" gossip_round={gossip_rounds}" if (step + 1) % sync_interval == 0 else ""
            print(
                f"decentralized train: step={step + 1}/{total_steps} "
                f"source={episode.source_id} mean_loss={sum(step_losses) / len(step_losses):.4f} "
                f"elapsed={time.perf_counter() - start_time:.1f}s{gossip_note}",
                flush=True,
            )
    elapsed_seconds = time.perf_counter() - start_time
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_decentralized_latent_checkpoint(
        output / "distributed_latent_decentralized.pt", nodes=nodes, model_name=model_name,
        route={"num_agents": len(nodes), "R": 4, "k": 2, "max_fanout": 2,
               "pilot_hospital_ids": list(pilot_ids),
               "graph_edges": [list(edge) for edge in graph.edge_list()],
               "split_paths": dict(split_paths)},
        gossip={"rounds": gossip_rounds, "sync_interval": sync_interval,
                "mixing_weights": {agent_id: dict(row) for agent_id, row in mixing_weights.items()}},
        training={"mode": "decentralized", "epochs": epochs, "batch_size": batch_size,
                  "local_steps": local_steps, "total_steps": total_steps,
                  "sync_interval": sync_interval, "seed": seed, "updates": len(losses),
                  "role_loss_weights": {"leaf": 0.5, "relay": 0.5, "source": 1.0},
                  "last_loss": losses[-1] if losses else float("nan"),
                  "elapsed_seconds": elapsed_seconds},
    )
    return {"updates": float(len(losses)), "last_loss": losses[-1] if losses else float("nan"),
            "elapsed_seconds": elapsed_seconds, "peak_memory_bytes": 0.0,
            "gossip_rounds": float(gossip_rounds)}


LatentKVProtocol = DistributedLatentProtocol

__all__ = [
    "KVBlock", "DistributedLatentProtocol", "DecentralizedLatentNode", "NodeLocalLatentTrainer", "LatentKVProtocol",
    "merge_kv_blocks", "slice_kv_block", "select_kv_rows", "detach_kv_block", "kv_wire_bytes",
    "batched_rollout", "batched_relay_aggregate",
    "accumulation_steps_for",
    "ring_two_hop_branches", "graph_two_hop_branches", "build_two_hop_route", "canonical_two_hop_losses", "graph_broadcast_tree",
    "metropolis_mixing_weights", "gossip_node_interfaces", "interface_consensus_distance",
    "save_distributed_latent_checkpoint", "load_distributed_latent_checkpoint",
    "save_decentralized_latent_checkpoint", "load_decentralized_latent_checkpoint",
]
