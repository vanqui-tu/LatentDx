import argparse
import json

import pytest

from scripts.run_distributed_medical import _build_graph, _load_graph_from_file, _summary


def _graph_args(*, num_agents=5, topology="ring", topology_file=None, seed=42, num_shortcuts=1):
    return argparse.Namespace(
        num_agents=num_agents,
        topology=topology,
        topology_file=topology_file,
        seed=seed,
        num_shortcuts=num_shortcuts,
    )


def test_shortcut_ring_topology_is_available_to_baseline_runner():
    graph = _build_graph(_graph_args(topology="shortcut_ring"))

    assert graph.edge_list() == ((0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (3, 4))


def test_topology_file_restores_m4_summary_edges(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "num_agents": 5,
        "graph_edges": [[0, 1], [0, 4], [1, 2], [1, 4], [2, 3], [3, 4]],
    }), encoding="utf-8")

    graph = _build_graph(_graph_args(topology_file=summary))

    assert graph.edge_list() == ((0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (3, 4))


def test_topology_file_rejects_incompatible_agent_count_and_duplicate_edges(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"num_agents": 5, "graph_edges": [[0, 1], [1, 0]]}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate edge"):
        _load_graph_from_file(summary, 5)
    with pytest.raises(ValueError, match="does not match"):
        _load_graph_from_file(summary, 10)


def test_summary_records_graph_for_a_follow_up_baseline():
    graph = _build_graph(_graph_args(topology="shortcut_ring"))

    summary = _summary([], (), {}, (), graph)

    assert summary["num_agents"] == 5
    assert summary["graph_edges"] == [[0, 1], [0, 4], [1, 2], [1, 4], [2, 3], [3, 4]]
