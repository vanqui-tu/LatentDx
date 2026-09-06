import json
from collections import Counter
from pathlib import Path

import pytest

from medlatent.distributed import (
    CommunicationGraph,
    MedicalBaselineKind,
    MedicalEpisode,
    MedicalQuery,
    MedicalQueryRecord,
    SynchronousEpisodeEngine,
    build_medical_agents,
    load_hospital_private_stores,
    load_medical_dataset_splits,
    load_medical_split,
    path_graph,
    run_medical_baseline,
    sample_balanced_sources,
)


def _row(case_id: str, hpo_code: str, disease: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "hpo_codes": [hpo_code],
        "phenotype_text": f"phenotype {case_id}",
        "disease_name": disease,
    }


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data))


def _medical_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    hospital_dir = tmp_path / "hospitals"
    hospital_dir.mkdir()
    _write_json(hospital_dir / "hospital_0.json", [_row("h0-match", "HP:1", "Disease zero")])
    _write_json(hospital_dir / "hospital_1.json", [_row("h1-match", "HP:2", "Disease one")])
    embeddings = tmp_path / "embeddings.json"
    ic = tmp_path / "ic.json"
    _write_json(embeddings, {"HP:1": [1.0, 0.0], "HP:2": [0.0, 1.0]})
    _write_json(ic, {"HP:1": 1.0, "HP:2": 1.0})
    return hospital_dir, embeddings, ic


def test_hospital_stores_do_not_retrieve_until_their_agent_uses_its_private_store(tmp_path: Path):
    hospital_dir, embeddings, ic = _medical_files(tmp_path)
    stores = load_hospital_private_stores(
        hospital_dir,
        num_agents=2,
        hpo_embeddings_file=embeddings,
        hpo_ic_file=ic,
    )
    agents = build_medical_agents(stores)
    query = load_medical_split(_write_split(tmp_path, "test", [_row("query", "HP:1", "Target")]))[0].query

    assert all(not store.retrieval_events for store in stores.values())
    retrieved = []

    def handler(agent, state, round_index):
        retrieved.extend(agent.retrieve_active_episode(state.episode_id, limit=1))
        state.deactivate()
        return ()

    SynchronousEpisodeEngine(path_graph(2), agents, max_rounds=1, max_fanout=1).run(
        episode_id="medical-episode",
        source_id=0,
        query=query,
        handler=handler,
    )

    assert retrieved[0].label == "Disease zero"
    with pytest.raises(RuntimeError, match="activated"):
        agents[0].retrieve_active_episode("medical-episode", limit=1)
    assert [event.agent_id for event in stores[0].retrieval_events] == [0]
    assert stores[1].retrieval_events == ()
    assert not hasattr(stores[0], "records")


def test_medical_split_checks_leakage_and_balances_deterministic_sources(tmp_path: Path):
    train = _write_split(tmp_path, "train", [_row(f"train-{index}", "HP:1", "Train") for index in range(3)])
    validation = _write_split(tmp_path, "validation", [_row(f"validation-{index}", "HP:1", "Validation") for index in range(3)])
    test = _write_split(tmp_path, "test", [_row(f"test-{index}", "HP:2", "Test") for index in range(7)])

    splits = load_medical_dataset_splits(train_file=train, validation_file=validation, test_file=test)
    first = sample_balanced_sources(splits.test, split="test", num_agents=3, seed=42)
    second = sample_balanced_sources(tuple(reversed(splits.test)), split="test", num_agents=3, seed=42)

    assert Counter(episode.source_id for episode in first) == Counter({0: 3, 1: 2, 2: 2})
    assert {episode.query.case_id: episode.source_id for episode in first} == {
        episode.query.case_id: episode.source_id for episode in second
    }
    assert all(not hasattr(episode.query, "target_label") for episode in first)
    assert {episode.target_label for episode in first} == {"Test"}


def test_medical_split_rejects_case_ids_shared_by_evaluation_splits(tmp_path: Path):
    train = _write_split(tmp_path, "train", [_row("shared", "HP:1", "Train")])
    validation = _write_split(tmp_path, "validation", [_row("validation", "HP:1", "Validation")])
    test = _write_split(tmp_path, "test", [_row("shared", "HP:2", "Test")])

    with pytest.raises(ValueError, match="overlap"):
        load_medical_dataset_splits(train_file=train, validation_file=validation, test_file=test)


def test_structured_medical_b0_to_b4_share_engine_and_emit_failure_stages(tmp_path: Path):
    hospital_dir, embeddings, ic = _medical_files(tmp_path)
    _write_json(hospital_dir / "hospital_2.json", [_row("h2-match", "HP:1", "Disease two")])
    stores = load_hospital_private_stores(
        hospital_dir,
        num_agents=3,
        hpo_embeddings_file=embeddings,
        hpo_ic_file=ic,
    )
    agents = build_medical_agents(stores)
    episode = MedicalEpisode("test", MedicalQuery("query", ("HP:1",), "phenotype query"), "Disease zero", 0)
    graph = path_graph(3)

    local = run_medical_baseline(episode, graph, agents, MedicalBaselineKind.LOCAL_ONLY, max_rounds=0, max_fanout=1)
    one_hop = run_medical_baseline(episode, graph, agents, MedicalBaselineKind.ONE_HOP, max_rounds=2, max_fanout=1)
    flooding = run_medical_baseline(episode, graph, agents, MedicalBaselineKind.FLOODING, max_rounds=4, max_fanout=2)
    random_k = run_medical_baseline(episode, graph, agents, MedicalBaselineKind.RANDOM_K, max_rounds=4, max_fanout=2, seed=42)
    heuristic = run_medical_baseline(
        episode,
        graph,
        agents,
        MedicalBaselineKind.HEURISTIC,
        max_rounds=4,
        max_fanout=2,
        public_expertise={1: ("HP:1",), 2: ("HP:2",)},
    )

    assert local.prediction == "Disease zero"
    assert one_hop.prediction == "Disease zero"
    assert flooding.prediction == "Disease zero"
    assert random_k.prediction == "Disease zero"
    assert heuristic.prediction == "Disease zero"
    assert all(not result.failure_stages for result in (local, one_hop, flooding, random_k, heuristic))
    assert all(any(event.stage == "retrieval" for event in result.trace) for result in (local, one_hop, flooding, random_k, heuristic))
    assert all(
        graph.has_edge(event.agent_id, event.receiver_id)
        for result in (one_hop, flooding, random_k, heuristic)
        for event in result.episode.events
        if event.event_type == "sent" and event.receiver_id is not None
    )


def test_medical_trace_marks_disconnected_topology_and_empty_retrieval(tmp_path: Path):
    hospital_dir, embeddings, ic = _medical_files(tmp_path)
    _write_json(hospital_dir / "hospital_2.json", [_row("h2-match", "HP:2", "Disease two")])
    agents = build_medical_agents(
        load_hospital_private_stores(
            hospital_dir,
            num_agents=3,
            hpo_embeddings_file=embeddings,
            hpo_ic_file=ic,
        )
    )
    episode = MedicalEpisode("test", MedicalQuery("query", ("HP:missing",), "phenotype query"), "Target", 0)
    disconnected = CommunicationGraph([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    result = run_medical_baseline(episode, disconnected, agents, MedicalBaselineKind.ONE_HOP, max_rounds=1, max_fanout=1)

    assert {event.stage for event in result.trace} >= {"unreachable", "routing", "retrieval", "aggregation"}


def _write_split(tmp_path: Path, name: str, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / f"{name}.json"
    _write_json(path, rows)
    return path
