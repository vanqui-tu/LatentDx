#!/usr/bin/env python
"""Run reproducible topology and budget sweeps for medical baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from dataclasses import asdict

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medlatent.distributed import load_hospital_private_stores, load_medical_split, run_topology_budget_sweep  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital_dir", required=True, type=Path)
    parser.add_argument("--split_file", required=True, type=Path)
    parser.add_argument("--hpo_embeddings_file", required=True, type=Path)
    parser.add_argument("--hpo_ic_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--agents", required=True, type=int)
    parser.add_argument("--topologies", default="complete,ring,star,path")
    parser.add_argument("--source_seeds", default="42")
    parser.add_argument("--node_assignment_seeds", default="0")
    parser.add_argument("--graph_seeds", default="42,43,44")
    parser.add_argument("--routing_seeds", default="0")
    parser.add_argument("--rounds", default="1,2,4")
    parser.add_argument("--fanouts", default="1,2")
    parser.add_argument("--random_regular_degree", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    stores = load_hospital_private_stores(
        args.hospital_dir,
        num_agents=args.agents,
        hpo_embeddings_file=args.hpo_embeddings_file,
        hpo_ic_file=args.hpo_ic_file,
    )
    records = load_medical_split(args.split_file)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    rows = run_topology_budget_sweep(
        records,
        stores,
        num_agents=args.agents,
        topology_kinds=_csv(args.topologies),
        source_seeds=_ints(args.source_seeds),
        node_assignment_seeds=_ints(args.node_assignment_seeds),
        graph_seeds=_ints(args.graph_seeds),
        routing_seeds=_ints(args.routing_seeds),
        rounds=_ints(args.rounds),
        fanouts=_ints(args.fanouts),
        random_regular_degree=args.random_regular_degree,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sweep.jsonl").write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {"rows": len(rows), "episodes": len(records), "topologies": _csv(args.topologies)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("CSV argument must not be empty")
    return values


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv(value))


if __name__ == "__main__":
    raise SystemExit(main())
