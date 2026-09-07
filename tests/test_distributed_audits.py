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


def _stores(tmp_path: Path, rows: list[list[dict[str, object]]] | None = None):
    hospital_dir = tmp_path / "hospitals"
    hospital_dir.mkdir()
    rows = rows or [
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
    assert audit.source_proxy_relevant is False
    assert audit.remote_proxy_relevant_agent_ids == (1,)
    assert audit.full_correct is True
    assert audit.collaboration_required is True
    assert audit.remote_necessary_agent_ids == (1,)
    assert dict(audit.removal_predictions)[2] == "Target"
    assert 0 not in dict(audit.removal_predictions)
    assert report.collaboration_required_rate == 1.0


def test_necessity_audit_separates_source_local_and_irrelevant_remote_agents(tmp_path: Path):
    stores = _stores(
        tmp_path,
        [
            [{"case_id": "source", "hpo_codes": ["HP:1"], "disease_name": "Target"}],
            [{"case_id": "other-1", "hpo_codes": ["HP:2"], "disease_name": "Other"}],
            [{"case_id": "other-2", "hpo_codes": ["HP:2"], "disease_name": "Other"}],
        ],
    )
    episode = MedicalEpisode("test", MedicalQuery("query", ("HP:1",), "phenotype"), "Target", 0)

    report = audit_medical_necessity(
        (episode,),
        path_graph(3),
        stores,
        baseline=MedicalBaselineKind.LOCAL_ONLY,
        max_rounds=0,
        max_fanout=2,
    )
    audit = report.episodes[0]

    assert audit.source_proxy_relevant is True
    assert audit.remote_proxy_relevant_agent_ids == ()
    assert audit.full_correct is True
    assert audit.collaboration_required is False
    assert audit.remote_necessary_agent_ids == ()
    assert audit.irrelevant_remote_removal_changed_prediction_ids == ()


def test_topology_budget_sweep_stratifies_evidence_distance_and_seeds(tmp_path: Path):
    stores = _stores(tmp_path)
    records = (MedicalQueryRecord(MedicalQuery("query", ("HP:1",), "phenotype"), "Target"),)
    first = run_topology_budget_sweep(
        records,
        stores,
        num_agents=3,
        topology_kinds=("ring",),
        source_seeds=(11,),
        graph_seeds=(42,),
        routing_seeds=(7,),
        rounds=(2,),
        fanouts=(2,),
        baselines=(MedicalBaselineKind.FLOODING,),
    )
    assert len(first) == 1
    assert first[0].graph_hash
    assert first[0].source_seed == 11
    assert first[0].node_assignment_seed == 0
    assert first[0].hospital_by_agent == ((0, 0), (1, 1), (2, 2))
    assert first[0].graph_seed == 42
    assert first[0].routing_seed == 7
    assert first[0].graph_edges == ((0, 1), (0, 2), (1, 2))
    assert first[0].episodes == 1
    assert first[0].evidence_distance == 1
    assert first[0].evidence_distance_stratum == "distance_1"
    assert first[0].mean_messages > 0.0

    remapped = run_topology_budget_sweep(
        records,
        stores,
        num_agents=3,
        topology_kinds=("ring",),
        source_seeds=(11,),
        node_assignment_seeds=(1,),
        graph_seeds=(42,),
        routing_seeds=(7,),
        rounds=(2,),
        fanouts=(2,),
        baselines=(MedicalBaselineKind.FLOODING,),
    )
    assert remapped[0].node_assignment_hash != first[0].node_assignment_hash
    assert remapped[0].hospital_by_agent != first[0].hospital_by_agent
