import json
from pathlib import Path

from medlatent.distributed import (
    MedicalBaselineKind,
    MedicalEpisode,
    MedicalQuery,
    MedicalQueryRecord,
    audit_medical_necessity,
    load_hospital_private_stores,
    path_graph,
    run_topology_budget_sweep,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def _stores(tmp_path: Path):
    hospital_dir = tmp_path / "hospitals"
    hospital_dir.mkdir()
    rows = [
        [{"case_id": "source", "hpo_codes": ["HP:2"], "disease_name": "Other"}],
        [{"case_id": "target", "hpo_codes": ["HP:1"], "disease_name": "Target"}],
        [{"case_id": "other", "hpo_codes": ["HP:2"], "disease_name": "Other"}],
    ]
    for agent_id, records in enumerate(rows):
        _write(hospital_dir / f"hospital_{agent_id}.json", records)
    embeddings = tmp_path / "embeddings.json"
    ic = tmp_path / "ic.json"
    _write(embeddings, {"HP:1": [1.0, 0.0], "HP:2": [0.0, 1.0]})
    _write(ic, {"HP:1": 1.0, "HP:2": 1.0})
    return load_hospital_private_stores(
        hospital_dir,
        num_agents=3,
        hpo_embeddings_file=embeddings,
        hpo_ic_file=ic,
    )


def test_necessity_audit_finds_target_agent_and_counterfactual_drop(tmp_path: Path):
    stores = _stores(tmp_path)
    episode = MedicalEpisode("test", MedicalQuery("query", ("HP:1",), "phenotype"), "Target", 0)

    report = audit_medical_necessity(
        (episode,),
        path_graph(3),
        stores,
        baseline=MedicalBaselineKind.FLOODING,
        max_rounds=4,
        max_fanout=2,
    )
    audit = report.episodes[0]

    assert audit.proxy_relevant_agent_ids == (1,)
    assert audit.full_correct is True
    assert audit.collaboration_required is True
    assert report.collaboration_required_rate == 1.0


def test_topology_budget_sweep_is_reproducible_and_reports_cost(tmp_path: Path):
    stores = _stores(tmp_path)
    records = (MedicalQueryRecord(MedicalQuery("query", ("HP:1",), "phenotype"), "Target"),)
    first = run_topology_budget_sweep(
        records,
        stores,
        num_agents=3,
        topology_kinds=("ring",),
        graph_seeds=(42,),
        rounds=(2,),
        fanouts=(2,),
        baselines=(MedicalBaselineKind.LOCAL_ONLY,),
    )
    assert len(first) == 1
    assert first[0].graph_hash
    assert first[0].episodes == 1
    assert first[0].mean_messages == 0.0
