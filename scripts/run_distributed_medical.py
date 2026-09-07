"""Run the fixed M2 medical communication comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medlatent.distributed import (  # noqa: E402
    MedicalBaselineKind,
    TransformersTextGenerator,
    build_medical_agents,
    load_hospital_private_stores,
    load_medical_split,
    ring_graph,
    run_medical_baseline,
    sample_balanced_sources,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hospital_dir", required=True, type=Path)
    parser.add_argument("--split_file", required=True, type=Path)
    parser.add_argument("--hpo_embeddings_file", required=True, type=Path)
    parser.add_argument("--hpo_ic_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--channel", choices=("structured", "text"), default="structured")
    parser.add_argument("--model_name")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    if args.channel == "text" and not args.model_name:
        parser.error("--model_name is required when --channel text")

    graph = ring_graph(5)
    stores = load_hospital_private_stores(
        args.hospital_dir, num_agents=5,
        hpo_embeddings_file=args.hpo_embeddings_file, hpo_ic_file=args.hpo_ic_file,
    )
    records = load_medical_split(args.split_file)
    if args.max_samples is not None:
        records = records[:args.max_samples]
    episodes = sample_balanced_sources(records, split=args.split_file.stem, num_agents=5, seed=args.seed)
    generator = (
        TransformersTextGenerator(args.model_name, device=args.device, dtype_name=args.dtype, local_files_only=args.local_files_only)
        if args.channel == "text" else None
    )
    rows = []
    for episode in episodes:
        for kind in MedicalBaselineKind:
            result = run_medical_baseline(
                episode, graph, build_medical_agents(stores), kind,
                max_rounds=4, max_fanout=2, channel=args.channel, text_generator=generator,
            )
            sent = sum(event.event_type == "sent" for event in result.episode.events)
            rows.append({
                "case_id": episode.query.case_id,
                "source_id": episode.source_id,
                "method": kind.value,
                "prediction": result.prediction,
                "target": episode.target_label,
                "contacted_agent_ids": list(result.contacted_agent_ids),
                "messages": sent,
                "wire_bytes": result.wire_bytes,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        method: {
            "accuracy": sum(row["prediction"] == row["target"] for row in rows if row["method"] == method) / max(1, sum(row["method"] == method for row in rows)),
            "mean_messages": sum(row["messages"] for row in rows if row["method"] == method) / max(1, sum(row["method"] == method for row in rows)),
            "mean_bytes": sum(row["wire_bytes"] for row in rows if row["method"] == method) / max(1, sum(row["method"] == method for row in rows)),
        }
        for method in (kind.value for kind in MedicalBaselineKind)
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
