"""Private hospital stores and deterministic medical episode sampling."""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent import AgentRuntime, PrivateKnowledgeStore
from ..retrieval import HpoCosineRetriever


@dataclass(frozen=True, slots=True)
class MedicalQuery:
    case_id: str
    hpo_codes: tuple[str, ...]
    phenotype_text: str


@dataclass(frozen=True, slots=True)
class MedicalQueryRecord:
    query: MedicalQuery
    target_label: str


@dataclass(frozen=True, slots=True)
class MedicalEpisode:
    split: str
    query: MedicalQuery
    target_label: str
    source_id: int


@dataclass(frozen=True, slots=True)
class MedicalRetrievedRecord:
    record_id: str
    label: str
    phenotype_text: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalAuditEvent:
    agent_id: int
    query_case_id: str
    result_count: int


class HospitalPrivateStore(PrivateKnowledgeStore[MedicalRetrievedRecord]):
    """One hospital shard, exposing only local derived retrieval results."""

    __slots__ = ("_agent_id", "__records", "_retriever", "__retrieval_events")

    def __init__(
        self,
        agent_id: int,
        records: Sequence[Mapping[str, object]],
        retriever: HpoCosineRetriever,
    ) -> None:
        if agent_id < 0:
            raise ValueError("agent_id must be non-negative")
        if not records:
            raise ValueError(f"hospital {agent_id} has no records")
        self._agent_id = agent_id
        self.__records = tuple(_normalize_record(record, index) for index, record in enumerate(records))
        self._retriever = retriever
        self.__retrieval_events: list[RetrievalAuditEvent] = []

    @property
    def agent_id(self) -> int:
        return self._agent_id

    @property
    def record_count(self) -> int:
        return len(self.__records)

    @property
    def retrieval_events(self) -> tuple[RetrievalAuditEvent, ...]:
        return tuple(self.__retrieval_events)

    def retrieve(self, query: object, *, limit: int | None = None) -> tuple[MedicalRetrievedRecord, ...]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        hpo_codes, case_id = _query_fields(query)
        ranked = self._retriever.rank(hpo_codes, self.__records, top_k=len(self.__records))
        matches = tuple(
            MedicalRetrievedRecord(
                record_id=str(result.record["case_id"]),
                label=str(result.record["disease_name"]),
                phenotype_text=str(result.record["phenotype_text"]),
                score=result.score,
            )
            for result in ranked
            if str(result.record["case_id"]) != case_id
        )
        selected = matches if limit is None else matches[:limit]
        self.__retrieval_events.append(RetrievalAuditEvent(self._agent_id, case_id, len(selected)))
        return selected


@dataclass(frozen=True, slots=True)
class MedicalDatasetSplits:
    train: tuple[MedicalQueryRecord, ...]
    validation: tuple[MedicalQueryRecord, ...]
    test: tuple[MedicalQueryRecord, ...]


def load_hospital_private_stores(
    hospital_dir: str | Path,
    *,
    num_agents: int,
    hpo_embeddings_file: str | Path,
    hpo_ic_file: str | Path,
) -> dict[int, HospitalPrivateStore]:
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    embeddings = _read_json_object(hpo_embeddings_file)
    ic_weights = _read_json_object(hpo_ic_file)
    root = Path(hospital_dir)
    return {
        agent_id: HospitalPrivateStore(
            agent_id,
            _read_json_list(root / f"hospital_{agent_id}.json"),
            HpoCosineRetriever(embeddings, ic_weights),
        )
        for agent_id in range(num_agents)
    }


def build_medical_agents(stores: Mapping[int, HospitalPrivateStore]) -> dict[int, AgentRuntime]:
    if not stores:
        raise ValueError("stores must not be empty")
    if set(stores) != set(range(len(stores))):
        raise ValueError("store IDs must be contiguous from zero")
    if any(store.agent_id != agent_id for agent_id, store in stores.items()):
        raise ValueError("store mapping keys must match store agent IDs")
    return {agent_id: AgentRuntime(agent_id, store) for agent_id, store in stores.items()}


def load_medical_split(path: str | Path) -> tuple[MedicalQueryRecord, ...]:
    records = tuple(
        MedicalQueryRecord(
            query=MedicalQuery(
                case_id=_case_id(row, index),
                hpo_codes=tuple(_hpo_codes(row)),
                phenotype_text=_phenotype_text(row),
            ),
            target_label=_disease_name(row),
        )
        for index, row in enumerate(_read_json_list(path))
    )
    _require_unique_case_ids(records, str(path))
    return records


def load_medical_dataset_splits(
    *,
    train_file: str | Path,
    validation_file: str | Path,
    test_file: str | Path,
) -> MedicalDatasetSplits:
    splits = MedicalDatasetSplits(
        train=load_medical_split(train_file),
        validation=load_medical_split(validation_file),
        test=load_medical_split(test_file),
    )
    named_splits = {"train": splits.train, "validation": splits.validation, "test": splits.test}
    for left_name, left_records in named_splits.items():
        left_ids = {record.query.case_id for record in left_records}
        for right_name, right_records in named_splits.items():
            if left_name >= right_name:
                continue
            overlap = left_ids.intersection(record.query.case_id for record in right_records)
            if overlap:
                raise ValueError(f"case IDs overlap between {left_name} and {right_name}: {sorted(overlap)[0]}")
    return splits


def sample_balanced_sources(
    records: Sequence[MedicalQueryRecord],
    *,
    split: str,
    num_agents: int,
    seed: int,
) -> tuple[MedicalEpisode, ...]:
    if not split:
        raise ValueError("split must not be empty")
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    _require_unique_case_ids(records, split)
    ordered_ids = sorted(record.query.case_id for record in records)
    random.Random(seed).shuffle(ordered_ids)
    source_by_case_id = {case_id: index % num_agents for index, case_id in enumerate(ordered_ids)}
    return tuple(
        MedicalEpisode(split, record.query, record.target_label, source_by_case_id[record.query.case_id])
        for record in records
    )


def _read_json_list(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"expected a JSON list of objects in {path}")
    return data


def _read_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _normalize_record(row: Mapping[str, object], index: int) -> dict[str, object]:
    return {
        "case_id": _case_id(row, index),
        "hpo_codes": _hpo_codes(row),
        "disease_name": _disease_name(row),
        "phenotype_text": _phenotype_text(row),
    }


def _case_id(row: Mapping[str, object], index: int) -> str:
    return str(row.get("case_id") or row.get("id") or f"case-{index}")


def _hpo_codes(row: Mapping[str, object]) -> list[str]:
    values = row.get("Phenotype", row.get("hpo_codes", ()))
    return [str(value) for value in values] if isinstance(values, Sequence) and not isinstance(values, str) else []


def _phenotype_text(row: Mapping[str, object]) -> str:
    names = row.get("phenotype_names")
    if isinstance(names, Sequence) and not isinstance(names, str) and names:
        return ", ".join(str(name) for name in names)
    return str(row.get("phenotypes_str") or row.get("phenotype_text") or ", ".join(_hpo_codes(row)))


def _disease_name(row: Mapping[str, object]) -> str:
    names = row.get("disease_names")
    if isinstance(names, Sequence) and not isinstance(names, str) and names:
        return str(names[0])
    value = row.get("diseases_prompt") or row.get("diseases_str") or row.get("disease_name")
    if value:
        return str(value)
    codes = row.get("RareDisease", row.get("disease_codes", ()))
    if isinstance(codes, Sequence) and not isinstance(codes, str) and codes:
        return str(codes[0])
    return "Unknown disease"


def _query_fields(query: object) -> tuple[tuple[str, ...], str]:
    if not isinstance(query, MedicalQuery):
        raise TypeError("medical store requires a MedicalQuery")
    return query.hpo_codes, query.case_id


def _require_unique_case_ids(records: Sequence[MedicalQueryRecord], scope: str) -> None:
    case_ids = [record.query.case_id for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"duplicate case IDs in {scope}")
