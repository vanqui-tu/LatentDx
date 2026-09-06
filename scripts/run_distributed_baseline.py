"""Run CPU-only synthetic distributed baselines B0-B2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medlatent.distributed import (  # noqa: E402
    BaselineKind,
    DistributedBaselineConfig,
    EpisodeLogRecord,
    complementary_evidence_fixture,
    run_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    config = DistributedBaselineConfig.from_dict(_read_config(args.config))
    graph = config.build_graph()
    fixture = complementary_evidence_fixture(graph)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph.save(args.output_dir / "graph.json")
    records: list[EpisodeLogRecord] = []
    for baseline in BaselineKind:
        episode_id = f"{config.config_hash[:12]}-{baseline.value}"
        result = run_baseline(
            fixture,
            baseline,
            max_rounds=config.rounds,
            max_fanout=config.max_fanout,
            episode_id=episode_id,
        )
        records.append(EpisodeLogRecord.from_baseline(config, fixture, result, episode_id=episode_id))

    with (args.output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.to_json() + "\n")
    summary = {
        "config_hash": config.config_hash,
        "graph_hash": graph.graph_hash,
        "baselines": [
            {
                "baseline": record.baseline,
                "prediction": record.prediction,
                "target": record.target,
                "correct": record.prediction == record.target,
                "messages": record.messages,
                "wire_bytes": record.wire_bytes,
                "termination_reason": record.termination_reason,
            }
            for record in records
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def _read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise SystemExit("YAML config requires PyYAML; use JSON or install pyyaml") from error
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit("config root must be a mapping")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
