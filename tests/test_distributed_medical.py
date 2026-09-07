import json
from pathlib import Path

from medlatent.distributed import (
    MedicalBaselineKind, MedicalEpisode, MedicalQuery, build_medical_agents,
    TextGeneration, load_hospital_private_stores, path_graph, run_medical_baseline,
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
    assert flood.contacted_agent_ids == (0, 1, 2)
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
