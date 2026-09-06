#!/usr/bin/env python
"""Run TextMAS B0-B4 over private medical hospital shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medlatent.distributed import (  # noqa: E402
    MedicalBaselineKind,
    TransformersTextGenerator,
    build_medical_agents,
    complete_graph,
    load_hospital_private_stores,
    load_medical_split,
    path_graph,
    ring_graph,
    run_medical_baseline,
    sample_balanced_sources,
    star_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--hospital_dir", required=True, type=Path)
    parser.add_argument("--split_file", required=True, type=Path)
    parser.add_argument("--hpo_embeddings_file", required=True, type=Path)
    parser.add_argument("--hpo_ic_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--topology", choices=["complete", "path", "ring", "star"], default="ring")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max_fanout", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_text_tokens", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    graph = {"complete": complete_graph, "path": path_graph, "ring": ring_graph, "star": star_graph}[args.topology](args.agents)
    stores = load_hospital_private_stores(
        args.hospital_dir,
        num_agents=args.agents,
        hpo_embeddings_file=args.hpo_embeddings_file,
        hpo_ic_file=args.hpo_ic_file,
    )
    agents = build_medical_agents(stores)
    records = load_medical_split(args.split_file)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    episodes = sample_balanced_sources(records, split=args.split_file.stem, num_agents=args.agents, seed=args.seed)
    generator = TransformersTextGenerator(
        args.model_name,
        device=args.device,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    for episode in episodes:
        for kind in MedicalBaselineKind:
            result = run_medical_baseline(
                episode,
                graph,
                agents,
                kind,
                max_rounds=args.rounds,
                max_fanout=args.max_fanout,
                seed=args.seed,
                channel="text",
                text_generator=generator,
                max_text_tokens=args.max_text_tokens,
                max_text_completion_tokens=args.max_new_tokens,
                episode_id=f"{episode.split}:{episode.query.case_id}:{kind.value}",
            )
            output_rows.append(
                {
                    "split": episode.split,
                    "case_id": episode.query.case_id,
                    "source_id": episode.source_id,
                    "baseline": kind.value,
                    "prediction": result.prediction,
                    "target": episode.target_label,
                    "correct": result.prediction == episode.target_label,
                    "contacted_agent_ids": list(result.contacted_agent_ids),
                    "messages": sum(event.event_type == "sent" for event in result.episode.events),
                    "text_tokens": result.text_tokens,
                    "text_prompt_tokens": result.text_prompt_tokens,
                    "text_completion_tokens": result.text_completion_tokens,
                    "wire_bytes": result.wire_bytes,
                    "leaked_substrings": list(result.leaked_substrings),
                    "failure_stages": list(result.failure_stages),
                    "termination_reason": result.episode.termination_reason,
                    "events": [event.__dict__ if hasattr(event, "__dict__") else {"type": event.event_type, "round": event.round_index, "agent_id": event.agent_id, "receiver_id": event.receiver_id} for event in result.episode.events],
                }
            )
    (args.output_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    summary = {
        "episodes": len(episodes),
        "rows": len(output_rows),
        "accuracy": sum(bool(row["correct"]) for row in output_rows) / len(output_rows) if output_rows else 0.0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
