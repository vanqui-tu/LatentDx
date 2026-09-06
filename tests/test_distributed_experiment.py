import json

import pytest

from medlatent.distributed import (
    BaselineKind,
    DistributedBaselineConfig,
    EpisodeLogRecord,
    complementary_evidence_fixture,
    run_baseline,
)


def _config_data():
    return {
        "seed": 42,
        "agents": 3,
        "source_policy": "fixed",
        "source_id": 0,
        "topology": {"kind": "path"},
        "rounds": 4,
        "max_fanout": 2,
        "router": "flood_unvisited",
        "channel": "structured",
        "aggregation": "require_all_evidence",
    }


def test_strict_config_builds_graph_and_has_stable_hash():
    first = DistributedBaselineConfig.from_dict(_config_data())
    second = DistributedBaselineConfig.from_dict(dict(_config_data()))

    assert first.config_hash == second.config_hash
    assert first.build_graph().graph_hash == complementary_evidence_fixture().graph.graph_hash


@pytest.mark.parametrize(
    "change, message",
    [
        ({"unknown": True}, "unknown config"),
        ({"channel": "text"}, "unknown channel"),
        ({"topology": {"kind": "directed"}}, "unknown topology"),
        ({"source_id": 3}, "source_id"),
    ],
)
def test_strict_config_rejects_unknown_or_invalid_settings(change, message):
    data = _config_data()
    data.update(change)

    with pytest.raises(ValueError, match=message):
        DistributedBaselineConfig.from_dict(data)


def test_episode_log_schema_contains_required_metadata_without_private_records():
    config = DistributedBaselineConfig.from_dict(_config_data())
    fixture = complementary_evidence_fixture(config.build_graph())
    result = run_baseline(
        fixture,
        BaselineKind.FLOODING,
        max_rounds=config.rounds,
        max_fanout=config.max_fanout,
        episode_id="schema-fixture-1",
    )

    record = EpisodeLogRecord.from_baseline(config, fixture, result, episode_id="schema-fixture-1")
    serialized = record.to_json()
    data = json.loads(serialized)

    assert data["config_hash"] == config.config_hash
    assert data["graph_hash"] == fixture.graph.graph_hash
    assert data["prediction"] == fixture.target_label
    assert data["target"] == fixture.target_label
    assert data["selected_edges_by_round"]
    assert data["cost"]["messages"] > 0
    assert "private-a" not in serialized
    assert "private-b" not in serialized
    assert "signal-a" not in serialized
    assert "signal-b" not in serialized
