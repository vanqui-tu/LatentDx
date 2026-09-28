import json
from pathlib import Path

import numpy as np

from medlatent.distributed import (
    CommunicationGraph, MedicalBaselineKind, MedicalEpisode, MedicalQuery, build_medical_agents,
    TextGeneration, build_agent_retrieval_batch, load_hospital_private_stores, path_graph, prediction_matches_target, run_medical_baseline,
)


def _row(case_id, hpo, disease):
    return {"case_id": case_id, "hpo_codes": [hpo], "phenotype_text": f"phenotype {case_id}", "disease_name": disease}


def _write(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_b2_returns_two_hop_private_evidence_to_source(tmp_path):
    hospitals = tmp_path / "hospitals"
    hospitals.mkdir()
    _write(hospitals / "hospital_0.json", [_row("source", "HP:other", "Wrong zero")])
    _write(hospitals / "hospital_1.json", [_row("bridge", "HP:other", "Wrong one")])
    _write(hospitals / "hospital_2.json", [_row("remote", "HP:gold", "Target")])
    embeddings = tmp_path / "embeddings.json"
    ic = tmp_path / "ic.json"
    _write(embeddings, {"HP:other": [1.0, 0.0], "HP:gold": [0.0, 1.0]})
    _write(ic, {"HP:other": 1.0, "HP:gold": 1.0})
    agents = build_medical_agents(load_hospital_private_stores(
        hospitals, num_agents=3, hpo_embeddings_file=embeddings, hpo_ic_file=ic
    ))
    episode = MedicalEpisode("test", MedicalQuery("query", ("HP:gold",), "phenotype query"), "Target", 0)
    graph = path_graph(3)
    local = run_medical_baseline(episode, graph, agents, MedicalBaselineKind.LOCAL_ONLY, max_rounds=0, max_fanout=2)
    one_hop = run_medical_baseline(episode, graph, agents, MedicalBaselineKind.ONE_HOP, max_rounds=2, max_fanout=2)
    flood = run_medical_baseline(episode, graph, agents, MedicalBaselineKind.FLOODING, max_rounds=4, max_fanout=2)
    assert local.prediction is None
    assert one_hop.prediction is None
    assert flood.prediction == "Target"
    assert flood.agents_reached == 3
    assert flood.wire_bytes > 0


def test_text_baseline_uses_same_route_and_fake_aggregation(tmp_path):
    hospitals = tmp_path / "hospitals"
    hospitals.mkdir()
    for index in range(2):
        _write(hospitals / f"hospital_{index}.json", [_row(f"h{index}", "HP:1", f"Disease {index}")])
    embeddings, ic = tmp_path / "embeddings.json", tmp_path / "ic.json"
    _write(embeddings, {"HP:1": [1.0]})
    _write(ic, {"HP:1": 1.0})
    agents = build_medical_agents(load_hospital_private_stores(hospitals, num_agents=2, hpo_embeddings_file=embeddings, hpo_ic_file=ic))
    episode = MedicalEpisode("test", MedicalQuery("query", ("HP:1",), "phenotype"), "Disease 0", 0)
    class FakeGenerator:
        def generate(self, system_prompt, user_prompt, *, max_new_tokens):
            return TextGeneration("<answer>Disease 0</answer>", 3, 2)

    result = run_medical_baseline(
        episode, path_graph(2), agents, MedicalBaselineKind.ONE_HOP, max_rounds=2, max_fanout=1,
        channel="text", text_generator=FakeGenerator(),
    )
    assert result.prediction == "Disease 0"
    assert result.text_tokens > 0
    assert result.text_prompt_tokens > 0


def test_b2_rounds_bound_request_depth_and_aggregate_replies_up_the_tree(tmp_path):
    hospitals = tmp_path / "hospitals"
    hospitals.mkdir()
    for index in range(5):
        _write(hospitals / f"hospital_{index}.json", [_row(f"h{index}", "HP:1", f"Disease {index}")])
    embeddings, ic = tmp_path / "embeddings.json", tmp_path / "ic.json"
    _write(embeddings, {"HP:1": [1.0]})
    _write(ic, {"HP:1": 1.0})
    agents = build_medical_agents(load_hospital_private_stores(
        hospitals, num_agents=5, hpo_embeddings_file=embeddings, hpo_ic_file=ic,
    ))
    episode = MedicalEpisode("test", MedicalQuery("query", ("HP:1",), "phenotype"), "Disease 4", 0)

    result = run_medical_baseline(
        episode, path_graph(5), agents, MedicalBaselineKind.FLOODING,
        max_rounds=4, max_fanout=1, seed=42,
    )

    assert result.agents_reached == 5
    sent = [(event.agent_id, event.receiver_id, event.message_id) for event in result.episode.events]
    assert all((left, right) in {(0, 1), (1, 2), (2, 3), (3, 4), (4, 3), (3, 2), (2, 1), (1, 0)}
               for left, right, _ in sent)
    assert not any(left == 4 and right == 0 for left, right, _ in sent)


def test_b2_claims_a_shared_child_once_per_query(tmp_path):
    hospitals = tmp_path / "hospitals"
    hospitals.mkdir()
    for index in range(4):
        _write(hospitals / f"hospital_{index}.json", [_row(f"h{index}", "HP:1", f"Disease {index}")])
    embeddings, ic = tmp_path / "embeddings.json", tmp_path / "ic.json"
    _write(embeddings, {"HP:1": [1.0]})
    _write(ic, {"HP:1": 1.0})
    agents = build_medical_agents(load_hospital_private_stores(
        hospitals, num_agents=4, hpo_embeddings_file=embeddings, hpo_ic_file=ic,
    ))
    graph = CommunicationGraph(np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ]))
    episode = MedicalEpisode("test", MedicalQuery("query", ("HP:1",), "phenotype"), "Disease 3", 0)

    result = run_medical_baseline(
        episode, graph, agents, MedicalBaselineKind.FLOODING,
        max_rounds=2, max_fanout=2, seed=42,
    )
    text_result = run_medical_baseline(
        episode, graph, agents, MedicalBaselineKind.FLOODING,
        max_rounds=2, max_fanout=2, channel="text", seed=42,
    )

    requests_to_shared_node = [
        event for event in result.episode.events
        if event.receiver_id == 3 and event.message_id is not None and ":request:" in event.message_id
    ]
    assert result.agents_reached == 4
    assert text_result.agents_reached == result.agents_reached
    assert len(requests_to_shared_node) == 1


def test_target_aliases_make_evaluation_language_and_format_insensitive():
    episode = MedicalEpisode(
        "test",
        MedicalQuery("query", ("HP:1",), "phenotype"),
        "Noonan 综合征",
        0,
        ("Noonan syndrome", "Noonan syndrome 1"),
    )

    assert prediction_matches_target("NOONAN-SYNDROME", episode)
    assert not prediction_matches_target("Costello syndrome", episode)


def test_agent_retrieval_batch_exposes_only_selected_store(tmp_path):
    hospitals = tmp_path / "hospitals"
    hospitals.mkdir()
    _write(hospitals / "hospital_0.json", [_row("local", "HP:1", "Local")])
    _write(hospitals / "hospital_1.json", [_row("private", "HP:2", "Private")])
    embeddings, ic = tmp_path / "embeddings.json", tmp_path / "ic.json"
    _write(embeddings, {"HP:1": [1.0, 0.0], "HP:2": [0.0, 1.0]})
    _write(ic, {"HP:1": 1.0, "HP:2": 1.0})
    stores = load_hospital_private_stores(
        hospitals, num_agents=2, hpo_embeddings_file=embeddings, hpo_ic_file=ic,
    )
    episodes = (MedicalEpisode("train", MedicalQuery("query", ("HP:1",), "phenotype"), "Local", 0),)
    batch = build_agent_retrieval_batch(episodes, stores, 0)
    assert batch.agent_id == 0
    assert batch.retrieved[0][0].label == "Local"
    assert not hasattr(batch, "stores")
