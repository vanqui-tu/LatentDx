#!/usr/bin/env python
"""Evaluate a frozen distributed latent checkpoint on the q4r6 pilot or expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medlatent.distributed import (  # noqa: E402
    DistributedLatentProtocol,
    CommunicationGraph,
    extend_graph,
    graph_two_hop_branches,
    load_hospital_private_stores,
    load_distributed_latent_checkpoint,
    load_medical_split,
    prediction_matches_target,
    sample_balanced_sources,
    select_kv_rows,
)
from medlatent.distributed.latent import _pilot_batch_tensors  # noqa: E402
from medlatent.hf_data import MedLatentDiagnosisDataset  # noqa: E402
from medlatent.modules import BoundaryEmbeddings, LatentDistiller  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(checkpoint_path: Path, model_name: str | None, device: torch.device, dtype: torch.dtype, local_files_only: bool):
    payload = load_distributed_latent_checkpoint(checkpoint_path, map_location="cpu")
    resolved_model = model_name or str(payload["model_name"])
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model, trust_remote_code=True, local_files_only=local_files_only, torch_dtype=dtype,
    ).to(device)
    distiller_state = payload["distiller"]
    boundary_state = payload["boundary"]
    hidden_size = int(distiller_state["hidden_size"])
    distiller = LatentDistiller(hidden_size).to(device=device, dtype=dtype)
    distiller.projection.load_state_dict(distiller_state["projection_state_dict"])
    distiller.latent_begin.data.copy_(distiller_state["latent_begin"].to(device=device, dtype=dtype))
    boundary = BoundaryEmbeddings(int(boundary_state["hidden_size"])).to(device=device, dtype=dtype)
    boundary.begin.data.copy_(boundary_state["begin"].to(device=device, dtype=dtype))
    boundary.end.data.copy_(boundary_state["end"].to(device=device, dtype=dtype))
    protocol = DistributedLatentProtocol(model, distiller, boundary, num_latents=int(payload["num_latents"]))
    protocol.distiller.eval()
    protocol.boundary.eval()
    return protocol, tokenizer, payload, resolved_model


def _prediction(tokenizer, token_ids: torch.Tensor) -> str:
    text = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True).strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    return (match.group(1) if match else text).strip()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _evaluate_method(*, method: str, protocol: DistributedLatentProtocol, tokenizer, episodes, rows_by_case,
                     graph, batch_size: int, pad_token_id: int, max_new_tokens: int, device: torch.device,
                     gold_agents: dict[str, set[int]]) -> tuple[list[dict[str, object]], float, int]:
    rows: list[dict[str, object]] = []
    total_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch = episodes[start:start + batch_size]
            tensors = _pilot_batch_tensors(
                episodes=batch, rows_by_case=rows_by_case, graph=graph,
                pad_token_id=pad_token_id, device=device,
            )
            started = time.perf_counter()
            if method == "local_only":
                branches = ()
                routes = [() for _ in batch]
                messages, latent_bytes = 0, 0
            else:
                routes = [graph_two_hop_branches(graph, episode.source_id) for episode in batch]
                leaf_blocks = protocol.rollout(tensors["leaf_ids"], tensors["leaf_mask"])
                if method == "latent_relay":
                    final_blocks = protocol.relay_rollout(
                        tensors["relay_ids"], tensors["relay_mask"], [leaf_blocks], detach_children=True,
                    )
                    messages = 4
                elif method == "latent_local":
                    final_blocks = leaf_blocks
                    messages = 2
                else:
                    raise ValueError(f"unknown method: {method}")
                branches = tuple(
                    select_kv_rows(final_blocks, list(range(branch, len(batch) * 2, 2)))
                    for branch in (0, 1)
                )
                one_branch = select_kv_rows(branches[0], 0)
                latent_bytes = messages * sum(
                    int(key.numel() * key.element_size() + value.numel() * value.element_size())
                    for key, value in one_branch
                ) + messages * 0 + 32 * messages
            generated = protocol.generate(
                tensors["source_ids"], tensors["source_mask"], branches,
                max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            total_seconds += time.perf_counter() - started
            for index, episode in enumerate(batch):
                prediction = _prediction(tokenizer, generated[index])
                useful = gold_agents.get(episode.query.case_id, set())
                if episode.source_id in useful:
                    stratum = "source_local"
                elif useful:
                    stratum = "remote"
                else:
                    stratum = "none"
                rows.append({
                    "case_id": episode.query.case_id,
                    "source_id": episode.source_id,
                    "method": method,
                    "prediction": prediction,
                    "target": episode.target_label,
                    "correct": prediction_matches_target(prediction, episode),
                    "evidence_stratum": stratum,
                    "contacted_agent_ids": sorted({node for route in routes[index] for node in route}),
                    "messages": messages,
                    "latent_bytes": latent_bytes,
                    "route": [list(route) for route in routes[index]],
                    "latency_seconds": (time.perf_counter() - started) / max(1, len(batch)),
                })
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return rows, total_seconds, peak


def _accuracy_by_stratum(rows: list[dict[str, object]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for stratum in ("source_local", "remote", "none"):
        selected = [row for row in rows if row["evidence_stratum"] == stratum]
        if selected:
            result[stratum] = sum(bool(row["correct"]) for row in selected) / len(selected)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--query_file", required=True, type=Path)
    parser.add_argument("--hospital_dir", required=True, type=Path)
    parser.add_argument("--hpo_embeddings_file", required=True, type=Path)
    parser.add_argument("--hpo_ic_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--hospital_ids", type=int, nargs="+", default=None)
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=["local_only", "latent_local", "latent_relay"],
                        choices=["local_only", "latent_local", "latent_relay"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_prompt_length", type=int, default=320)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_samples < 0:
        parser.error("batch_size must be positive and max_samples non-negative")
    payload = load_distributed_latent_checkpoint(args.checkpoint, map_location="cpu")
    route = payload.get("route", {})
    pilot_ids = tuple(int(value) for value in route.get("pilot_hospital_ids", (0, 1, 2, 3, 4)))
    selected_ids = tuple(args.hospital_ids or (pilot_ids if args.num_agents in (None, 5) else range(args.num_agents)))
    num_agents = args.num_agents or len(selected_ids)
    if len(selected_ids) != num_agents:
        parser.error("hospital_ids count must equal num_agents")
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    protocol, tokenizer, payload, model_name = _load_protocol(args.checkpoint, args.model_name, device, dtype, args.local_files_only)
    dataset = MedLatentDiagnosisDataset(
        data_file=args.query_file, hospital_dir=args.hospital_dir, tokenizer=tokenizer,
        num_hospitals=num_agents, hospital_ids=list(selected_ids),
        max_prompt_length=args.max_prompt_length, max_target_length=args.max_target_length,
        hpo_embeddings_file=args.hpo_embeddings_file, hpo_ic_file=args.hpo_ic_file,
    )
    source_agent_count = pilot_count = len(pilot_ids)
    episodes = sample_balanced_sources(
        load_medical_split(args.query_file), split=args.query_file.stem,
        num_agents=source_agent_count if num_agents > source_agent_count else num_agents,
        seed=args.seed,
    )
    if args.max_samples:
        episodes = episodes[:args.max_samples]
    rows_by_case = {row["case_id"]: row for row in dataset}
    graph_seed = int(route.get("graph_seed", 42))
    pilot_edges = tuple(tuple(int(value) for value in edge) for edge in route.get("graph_edges", ()))
    if len(pilot_edges) != 0:
        pilot_adjacency = torch.zeros((pilot_count, pilot_count), dtype=torch.bool).numpy()
        for left, right in pilot_edges:
            if left >= pilot_count or right >= pilot_count or left == right:
                parser.error("checkpoint pilot graph contains invalid edge")
            pilot_adjacency[left, right] = pilot_adjacency[right, left] = True
        pilot_graph = CommunicationGraph(pilot_adjacency)
    else:
        from medlatent.distributed import shortcut_ring_graph
        pilot_graph = shortcut_ring_graph(pilot_count, seed=graph_seed, num_shortcuts=1)
    graph = extend_graph(pilot_graph, num_agents, seed=graph_seed) if num_agents > pilot_count else pilot_graph
    stores = load_hospital_private_stores(
        args.hospital_dir, num_agents=num_agents, hospital_ids=selected_ids,
        hpo_embeddings_file=args.hpo_embeddings_file, hpo_ic_file=args.hpo_ic_file,
    )
    gold_agents: dict[str, set[int]] = {}
    for episode in episodes:
        gold_agents[episode.query.case_id] = {
            agent_id for agent_id, store in stores.items()
            if (records := store.retrieve(episode.query, limit=1))
            and prediction_matches_target(records[0].label, episode)
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, float]] = {}
    peak_memory = 0
    for method in args.methods:
        method_rows, elapsed, method_peak = _evaluate_method(
            method=method, protocol=protocol, tokenizer=tokenizer, episodes=episodes,
            rows_by_case=rows_by_case, graph=graph, batch_size=args.batch_size,
            pad_token_id=tokenizer.pad_token_id, max_new_tokens=args.max_new_tokens,
            device=device, gold_agents=gold_agents,
        )
        _write_jsonl(args.output_dir / f"{method}.jsonl", method_rows)
        count = len(method_rows)
        summaries[method] = {
            "accuracy": sum(bool(row["correct"]) for row in method_rows) / count if count else 0.0,
            "mean_messages": sum(int(row["messages"]) for row in method_rows) / count if count else 0.0,
            "mean_latent_bytes": sum(int(row["latent_bytes"]) for row in method_rows) / count if count else 0.0,
            "mean_latency_seconds": elapsed / count if count else 0.0,
            "accuracy_by_evidence_stratum": _accuracy_by_stratum(method_rows),
        }
        peak_memory = max(peak_memory, method_peak)
    summary = {
        "model_name": model_name, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint), "query_file": str(args.query_file),
        "hospital_dir": str(args.hospital_dir), "num_agents": num_agents,
        "hospital_ids": list(selected_ids), "pilot_hospital_ids": list(pilot_ids),
        "source_assignment_agents": source_agent_count if num_agents > source_agent_count else num_agents,
        "graph_seed": graph_seed, "graph_edges": [list(edge) for edge in graph.edge_list()],
        "pilot_graph_edges": [list(edge) for edge in pilot_graph.edge_list()],
        "graph_extension_preserves_pilot": set(pilot_graph.edge_list()).issubset(set(graph.edge_list())),
        "R": int(route.get("R", 4)), "k": int(route.get("k", 2)),
        "m": int(payload["num_latents"]), "episodes": len(episodes),
        "peak_memory_bytes": peak_memory, "methods": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"distributed latent evaluation complete: output_dir={args.output_dir} episodes={len(episodes)}")


if __name__ == "__main__":
    main()
