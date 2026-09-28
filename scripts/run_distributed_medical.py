"""Run reproducible fixed-topology M2 medical communication baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

# cuBLAS reads this before its first CUDA operation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medlatent.distributed import (  # noqa: E402
    CommunicationGraph,
    MedicalBaselineKind,
    TransformersTextGenerator,
    VllmTextGenerator,
    build_medical_agents,
    complete_graph,
    load_hospital_private_stores,
    load_medical_split,
    path_graph,
    prediction_matches_target,
    ring_graph,
    run_medical_baseline,
    sample_balanced_sources,
    shortcut_ring_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital_dir", required=True, type=Path)
    parser.add_argument("--split_file", required=True, type=Path)
    parser.add_argument("--hpo_embeddings_file", required=True, type=Path)
    parser.add_argument("--hpo_ic_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--channel", choices=("structured", "text"), default="structured")
    parser.add_argument("--methods", choices=tuple(kind.value for kind in MedicalBaselineKind), nargs="+", default=[kind.value for kind in MedicalBaselineKind])
    parser.add_argument("--model_name")
    parser.add_argument("--vllm_base_url")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_agents", type=int, default=5)
    parser.add_argument(
        "--topology",
        choices=("ring", "shortcut_ring", "shortcut-ring", "complete", "path"),
        default="ring",
        help="Generated topology. Ignored when --topology_file is supplied.",
    )
    parser.add_argument(
        "--topology_file",
        type=Path,
        help="Existing JSON summary or run manifest containing graph_edges or edge_list.",
    )
    parser.add_argument(
        "--num_shortcuts",
        type=int,
        default=1,
        help="Number of seeded non-ring edges for --topology shortcut_ring (default: 1).",
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max_fanout", type=int, default=2)
    parser.add_argument("--max_text_tokens", type=int, default=128)
    parser.add_argument("--max_text_wire_bytes", type=int, default=4_096)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    _validate_args(parser, args)
    _seed_everything(args.seed, deterministic=args.deterministic)
    methods = tuple(MedicalBaselineKind(value) for value in args.methods)

    try:
        graph = _build_graph(args)
    except ValueError as error:
        parser.error(str(error))
    stores = load_hospital_private_stores(
        args.hospital_dir,
        num_agents=args.num_agents,
        hpo_embeddings_file=args.hpo_embeddings_file,
        hpo_ic_file=args.hpo_ic_file,
    )
    records = load_medical_split(args.split_file)
    if args.max_samples is not None:
        records = records[:args.max_samples]
    episodes = sample_balanced_sources(records, split=args.split_file.stem, num_agents=args.num_agents, seed=args.seed)
    generator = (
        VllmTextGenerator(
            args.model_name,
            base_url=args.vllm_base_url,
        )
        if args.channel == "text" and args.vllm_base_url
        else TransformersTextGenerator(args.model_name, device=args.device, dtype_name=args.dtype, local_files_only=args.local_files_only)
        if args.channel == "text"
        else None
    )

    rows: list[dict[str, object]] = []
    for episode in episodes:
        for kind in methods:
            result = run_medical_baseline(
                episode,
                graph,
                build_medical_agents(stores),
                kind,
                max_rounds=args.rounds,
                max_fanout=args.max_fanout,
                channel=args.channel,
                max_text_tokens=args.max_text_tokens,
                max_text_wire_bytes=args.max_text_wire_bytes,
                text_generator=generator,
                max_text_completion_tokens=args.max_new_tokens,
            )
            rows.append({
                "case_id": episode.query.case_id,
                "source_id": episode.source_id,
                "method": kind.value,
                "prediction": result.prediction,
                "target": episode.target_label,
                "contacted_agent_ids": list(result.contacted_agent_ids),
                "messages": sum(event.event_type == "sent" for event in result.episode.events),
                "wire_bytes": result.wire_bytes,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "episodes.jsonl", rows)
    summary = _summary(rows, episodes, stores, methods, graph)
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "run.json", _manifest(args, graph))
    print(json.dumps(summary, sort_keys=True))
    return 0


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.channel == "text" and not args.model_name:
        parser.error("--model_name is required when --channel text")
    if args.vllm_base_url and args.channel != "text":
        parser.error("--vllm_base_url requires --channel text")
    for name in ("num_agents", "rounds", "max_fanout", "max_text_tokens", "max_text_wire_bytes", "max_new_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.num_shortcuts < 0:
        parser.error("--num_shortcuts must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")


def _build_graph(args: argparse.Namespace) -> CommunicationGraph:
    """Use an explicit saved graph when available, otherwise generate one."""
    if args.topology_file is not None:
        return _load_graph_from_file(args.topology_file, args.num_agents)
    if args.topology in {"shortcut_ring", "shortcut-ring"}:
        return shortcut_ring_graph(
            args.num_agents,
            seed=args.seed,
            num_shortcuts=args.num_shortcuts,
        )
    return {"ring": ring_graph, "complete": complete_graph, "path": path_graph}[args.topology](args.num_agents)


def _load_graph_from_file(path: Path, num_agents: int) -> CommunicationGraph:
    """Restore an undirected graph recorded by a latent summary or baseline manifest."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        raise ValueError(f"could not read topology file: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"topology file is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("topology file must contain a JSON object")

    recorded_agents = payload.get("num_agents")
    if recorded_agents is not None and (
        isinstance(recorded_agents, bool)
        or not isinstance(recorded_agents, int)
        or recorded_agents != num_agents
    ):
        raise ValueError(
            f"topology file num_agents={recorded_agents!r} does not match --num_agents={num_agents}"
        )
    if "graph_edges" in payload:
        raw_edges = payload["graph_edges"]
    elif "edge_list" in payload:
        raw_edges = payload["edge_list"]
    else:
        raise ValueError("topology file must contain graph_edges or edge_list")
    if not isinstance(raw_edges, list):
        raise ValueError("topology edges must be a JSON array")

    adjacency = np.zeros((num_agents, num_agents), dtype=np.bool_)
    for raw_edge in raw_edges:
        if (
            not isinstance(raw_edge, list)
            or len(raw_edge) != 2
            or any(isinstance(node, bool) or not isinstance(node, int) for node in raw_edge)
        ):
            raise ValueError("each topology edge must be a two-element integer array")
        left, right = raw_edge
        if not 0 <= left < num_agents or not 0 <= right < num_agents:
            raise ValueError(f"topology edge has node outside [0, {num_agents}): {raw_edge!r}")
        if adjacency[left, right] or adjacency[right, left]:
            raise ValueError(f"topology file contains duplicate edge: {raw_edge!r}")
        adjacency[left, right] = adjacency[right, left] = True
    return CommunicationGraph(adjacency)


def _seed_everything(seed: int, *, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def _summary(
    rows: list[dict[str, object]],
    episodes: tuple,
    stores: dict,
    methods: tuple[MedicalBaselineKind, ...],
    graph: CommunicationGraph,
) -> dict[str, object]:
    episode_by_case_id = {episode.query.case_id: episode for episode in episodes}
    method_summary: dict[str, dict[str, float]] = {}
    for method in (kind.value for kind in methods):
        method_rows = [row for row in rows if row["method"] == method]
        count = len(method_rows)
        method_summary[method] = {
            "accuracy": sum(prediction_matches_target(row["prediction"], episode_by_case_id[row["case_id"]]) for row in method_rows) / count,
            "mean_messages": sum(int(row["messages"]) for row in method_rows) / count,
            "mean_bytes": sum(int(row["wire_bytes"]) for row in method_rows) / count,
        }
    return {
        "num_agents": graph.num_agents,
        "graph_edges": [list(edge) for edge in graph.edge_list()],
        "episodes": len(episodes),
        "methods": method_summary,
        "retrieval_coverage": _retrieval_coverage(episodes, stores),
    }


def _retrieval_coverage(episodes: tuple, stores: dict) -> dict[str, float]:
    source_local = remote = no_gold_anywhere = 0
    for episode in episodes:
        gold_agents = {
            agent_id
            for agent_id, store in stores.items()
            if (records := store.retrieve(episode.query, limit=1)) and prediction_matches_target(records[0].label, episode)
        }
        source_local += episode.source_id in gold_agents
        remote += bool(gold_agents.difference({episode.source_id}))
        no_gold_anywhere += not gold_agents
    count = len(episodes)
    return {
        "source_local_gold_hit_fraction": source_local / count if count else 0.0,
        "remote_gold_hit_fraction": remote / count if count else 0.0,
        "no_gold_hit_anywhere_fraction": no_gold_anywhere / count if count else 0.0,
    }


def _manifest(args: argparse.Namespace, graph) -> dict[str, object]:
    return {
        "command": [sys.executable, *sys.argv],
        "git_commit": _git_commit(),
        "channel": args.channel,
        "model_name": args.model_name,
        "text_backend": "vllm" if args.vllm_base_url else ("transformers" if args.channel == "text" else None),
        "vllm_base_url": args.vllm_base_url,
        "target_match_rule": "casefolded punctuation-insensitive match against target_label or disease_aliases",
        "methods": args.methods,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "num_agents": args.num_agents,
        "topology": "file" if args.topology_file is not None else args.topology.replace("-", "_"),
        "topology_file": str(args.topology_file) if args.topology_file is not None else None,
        "topology_file_sha256": _sha256(args.topology_file) if args.topology_file is not None else None,
        "num_shortcuts": args.num_shortcuts if args.topology_file is None else None,
        "edge_list": [list(edge) for edge in graph.edge_list()],
        "rounds": args.rounds,
        "max_fanout": args.max_fanout,
        "max_samples": args.max_samples,
        "max_text_tokens": args.max_text_tokens,
        "max_text_wire_bytes": args.max_text_wire_bytes,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "input_sha256": {
            name: _sha256(getattr(args, name))
            for name in ("split_file", "hpo_embeddings_file", "hpo_ic_file")
        },
        "hospital_sha256": {
            f"hospital_{agent_id}.json": _sha256(args.hospital_dir / f"hospital_{agent_id}.json")
            for agent_id in range(args.num_agents)
        },
        "library_versions": _library_versions(),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "numpy"):
        try:
            module = __import__(name)
            versions[name] = str(module.__version__)
        except ImportError:
            versions[name] = None
    return versions


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
