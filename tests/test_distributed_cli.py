import json
import subprocess
import sys


def test_distributed_baseline_cli_writes_reproducible_smoke_outputs(tmp_path):
    command = [
        sys.executable,
        "scripts/run_distributed_baseline.py",
        "--config",
        "configs/distributed_baseline.yaml",
        "--output_dir",
        str(tmp_path / "smoke"),
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    first_dir = tmp_path / "smoke"
    first_summary = json.loads((first_dir / "summary.json").read_text(encoding="utf-8"))
    first_graph = (first_dir / "graph.json").read_text(encoding="utf-8")
    first_jsonl = (first_dir / "episodes.jsonl").read_text(encoding="utf-8")

    second_dir = tmp_path / "smoke-second"
    subprocess.run(command[:-1] + [str(second_dir)], capture_output=True, text=True, check=True)
    second_summary = json.loads((second_dir / "summary.json").read_text(encoding="utf-8"))

    assert first_summary == second_summary
    assert first_graph == (second_dir / "graph.json").read_text(encoding="utf-8")
    assert first_jsonl == (second_dir / "episodes.jsonl").read_text(encoding="utf-8")
    assert {row["baseline"] for row in first_summary["baselines"]} == {"B0", "B1", "B2"}
    assert first_summary["baselines"][-1]["correct"] is True
