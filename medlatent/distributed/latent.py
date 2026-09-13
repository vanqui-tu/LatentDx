"""Differentiable fixed-route latent KV communication.

The protocol transports only a newly encoded ``(m + 2)`` position KV block:
``BEGIN``, ``m`` distilled positions, and ``END``.  KV blocks are tuples of
``(key, value)`` tensors, one pair per decoder layer, shaped
``[batch, heads, positions, head_dim]``.  Blocks are concatenated in stable
child/branch order before a relay re-encodes them with its private prompt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from ..hf_data import IGNORE_INDEX
from ..hf_medlatent_h import _build_position_ids, _iter_key_value_pairs
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
        logits = self._source_sequence_logits(input_ids, attention_mask, branch_blocks)
        last = attention_mask.sum(dim=1).long() - 1
        return logits[torch.arange(input_ids.shape[0], device=input_ids.device), last]

    def source_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    target_ids: torch.Tensor, target_labels: torch.Tensor,
                    branch_blocks: Sequence[KVBlock] = ()) -> torch.Tensor:
        """Teacher-force a diagnosis after the source prompt and branch blocks."""
        if target_ids.ndim != 2 or target_labels.shape != target_ids.shape:
            raise ValueError("target_ids and target_labels must both be [batch, sequence]")
        if input_ids.shape[0] != target_ids.shape[0]:
            raise ValueError("source and target batches must have the same size")
        target_mask = (target_labels != IGNORE_INDEX).long()
        full_ids = torch.cat([input_ids, target_ids], dim=1)
        full_mask = torch.cat([attention_mask, target_mask], dim=1)
        logits = self._source_sequence_logits(full_ids, full_mask, branch_blocks)
        source_width = input_ids.shape[1]
        first = logits[:, source_width - 1, :]
        target_logits = logits[:, source_width:source_width + target_ids.shape[1], :]
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
        return logits


def two_hop_ring_blocks(protocol: DistributedLatentProtocol, *, leaf_ids: torch.Tensor,
                        leaf_mask: torch.Tensor, relay_ids: torch.Tensor,
                        relay_mask: torch.Tensor) -> tuple[KVBlock, KVBlock]:
    """Return the two final branch summaries from batched leaf/relay passes."""
    if leaf_ids.shape[0] != 2 or relay_ids.shape[0] != 2:
        raise ValueError("leaf and relay batches must contain the two ring branches")
    leaf_blocks = protocol.rollout(leaf_ids, leaf_mask)
    relay_blocks = protocol.rollout(relay_ids, relay_mask, [leaf_blocks])
    return select_kv_rows(relay_blocks, 0), select_kv_rows(relay_blocks, 1)


def run_two_hop_ring_query(protocol: DistributedLatentProtocol, *, source_ids: torch.Tensor,
                           source_mask: torch.Tensor, leaf_ids: torch.Tensor, leaf_mask: torch.Tensor,
                           relay_ids: torch.Tensor, relay_mask: torch.Tensor) -> torch.Tensor:
    """Run one query with two hospitals batched at each fixed ring hop."""
    branches = two_hop_ring_blocks(protocol, leaf_ids=leaf_ids, leaf_mask=leaf_mask,
                                   relay_ids=relay_ids, relay_mask=relay_mask)
    return protocol.source_logits(source_ids, source_mask, branches)


def save_distributed_latent_checkpoint(path: str | Path, *, protocol: DistributedLatentProtocol,
                                       model_name: str, route: Mapping[str, int], optimizer: torch.optim.Optimizer,
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


def load_distributed_latent_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if payload.get("module_type") != "DistributedLatentKV":
        raise ValueError(f"Expected DistributedLatentKV checkpoint, got {payload.get('module_type')!r}")
    return payload


def train_distributed_latent_real(*, model_name: str, train_file: str | Path,
                                  hospital_dir: str | Path, output_dir: str | Path,
                                  hpo_embeddings_file: str | Path, hpo_ic_file: str | Path,
                                  validation_file: str | Path | None = None,
                                  num_agents: int = 10, num_latents: int = 8,
                                  max_prompt_length: int = 320, max_target_length: int = 64,
                                  epochs: int = 1, max_steps: int = 0, learning_rate: float = 1e-4,
                                  weight_decay: float = 0.01, seed: int = 42,
                                  device: str = "cuda", dtype: str = "bfloat16",
                                  local_files_only: bool = False) -> dict[str, float]:
    """Train the fixed N=10 ring protocol one query at a time."""
    if num_agents != 10:
        raise ValueError("M3 fixed route requires num_agents=10")
    if epochs <= 0 or (max_steps < 0):
        raise ValueError("epochs must be positive and max_steps non-negative")
    torch.manual_seed(seed)
    resolved_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..hf_data import MedLatentDiagnosisDataset, _pad
    from ..medical import load_medical_split, sample_balanced_sources

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    ).to(resolved_device)
    protocol = DistributedLatentProtocol(
        model,
        LatentDistiller(int(model.config.hidden_size)).to(device=resolved_device, dtype=torch_dtype),
        BoundaryEmbeddings(int(model.config.hidden_size)).to(device=resolved_device, dtype=torch_dtype),
        num_latents=num_latents,
    )
    dataset = MedLatentDiagnosisDataset(
        data_file=train_file, hospital_dir=hospital_dir, tokenizer=tokenizer,
        num_hospitals=num_agents, max_prompt_length=max_prompt_length,
        max_target_length=max_target_length, hpo_embeddings_file=hpo_embeddings_file,
        hpo_ic_file=hpo_ic_file,
    )
    episodes = sample_balanced_sources(
        load_medical_split(train_file), split=Path(train_file).stem,
        num_agents=num_agents, seed=seed,
    )
    rows_by_case = {row["case_id"]: row for row in dataset}
    optimizer = torch.optim.AdamW(
        list(protocol.distiller.parameters()) + list(protocol.boundary.parameters()),
        lr=learning_rate, weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    updates = 0
    last_loss = float("nan")
    protocol.distiller.train()
    protocol.boundary.train()
    for epoch in range(epochs):
        order = torch.randperm(len(episodes), generator=generator).tolist()
        for position in order:
            episode = episodes[position]
            row = rows_by_case[episode.query.case_id]
            paths = ring_two_hop_branches(episode.source_id, num_agents)
            leaf_rows = _pad([row["hospital_ids_all"][leaf] for leaf, _ in paths], tokenizer.pad_token_id)
            relay_rows = _pad([row["hospital_ids_all"][relay] for _, relay in paths], tokenizer.pad_token_id)
            source_tokens = row["hospital_ids_all"][episode.source_id] + row["host_question_ids"]
            source_ids, source_mask = _pad([source_tokens], tokenizer.pad_token_id)
            target_ids, _ = _pad([row["target_ids"]], tokenizer.pad_token_id)
            target_labels = target_ids.clone()
            target_labels[target_ids == tokenizer.pad_token_id] = IGNORE_INDEX
            leaf_ids, leaf_mask = (item.to(resolved_device) for item in leaf_rows)
            relay_ids, relay_mask = (item.to(resolved_device) for item in relay_rows)
            source_ids, source_mask = source_ids.to(resolved_device), source_mask.to(resolved_device)
            target_ids, target_labels = target_ids.to(resolved_device), target_labels.to(resolved_device)
            branch_blocks = two_hop_ring_blocks(
                protocol, leaf_ids=leaf_ids, leaf_mask=leaf_mask,
                relay_ids=relay_ids, relay_mask=relay_mask,
            )
            loss = protocol.source_loss(
                source_ids, source_mask, target_ids, target_labels, branch_blocks,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(protocol.distiller.parameters()) + list(protocol.boundary.parameters()), 1.0,
            )
            optimizer.step()
            updates += 1
            last_loss = float(loss.detach().cpu())
            if max_steps and updates >= max_steps:
                break
        if max_steps and updates >= max_steps:
            break
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_distributed_latent_checkpoint(
        output / "distributed_latent_final.pt", protocol=protocol, model_name=model_name,
        route={"num_agents": num_agents, "rounds": 4, "max_fanout": 2}, optimizer=optimizer,
        training={"seed": seed, "epochs": epoch + 1, "updates": updates, "last_loss": last_loss},
    )
    (output / "training_summary.json").write_text(json.dumps({
        "model_name": model_name, "num_agents": num_agents, "num_latents": num_latents,
        "updates": updates, "last_loss": last_loss,
    }, indent=2) + "\n", encoding="utf-8")
    return {"updates": float(updates), "last_loss": last_loss}


LatentKVProtocol = DistributedLatentProtocol

__all__ = [
    "KVBlock", "DistributedLatentProtocol", "LatentKVProtocol",
    "merge_kv_blocks", "slice_kv_block", "select_kv_rows", "kv_wire_bytes",
    "ring_two_hop_branches", "two_hop_ring_blocks", "run_two_hop_ring_query",
    "save_distributed_latent_checkpoint", "load_distributed_latent_checkpoint",
]
