"""Dataset utilities for real MedLatent-H training/evaluation."""

from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .prompts import SYSTEM_PROMPT, build_hospital_prompt, build_host_question, build_target_answer
from .retrieval import HpoCosineRetriever

IGNORE_INDEX = -100


def _read_json(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8-sig") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return rows


def _read_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8-sig") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return obj


def _case_id(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("case_id") or row.get("id") or f"case-{fallback}")


def _hpo_codes(row: dict[str, Any]) -> list[str]:
    return [str(x) for x in row.get("Phenotype", row.get("hpo_codes", []))]


def _phenotype_text(row: dict[str, Any]) -> str:
    names = row.get("phenotype_names")
    if isinstance(names, list) and names:
        return ", ".join(str(x) for x in names)
    text = row.get("phenotypes_str") or row.get("phenotype_text")
    if text:
        return str(text)
    return ", ".join(_hpo_codes(row))


def _disease_name(row: dict[str, Any]) -> str:
    names = row.get("disease_names")
    if isinstance(names, list) and names:
        return str(names[0])
    text = row.get("diseases_prompt") or row.get("diseases_str") or row.get("disease_name")
    if text:
        return str(text)
    codes = row.get("RareDisease", row.get("disease_codes", []))
    if isinstance(codes, list) and codes:
        return str(codes[0])
    return "Unknown disease"


def _omim_code(row: dict[str, Any]) -> str:
    codes = row.get("RareDisease", row.get("disease_codes", []))
    if isinstance(codes, list) and codes:
        return str(codes[0])
    return str(row.get("omim_id") or row.get("disease_code") or _disease_name(row))


def _tokenize(tokenizer, text: str, max_len: int | None = None) -> list[int]:
    if max_len is not None:
        return tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _render_chat(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    except Exception:
        joined = "\n\n".join(f"{msg['role'].title()}: {msg['content']}" for msg in messages)
        return joined + ("\n\nAssistant:" if add_generation_prompt else "")


class MedLatentDiagnosisDataset(Dataset):
    """Prepare one query with one prompt per hospital and one answer target."""

    def __init__(
        self,
        *,
        data_file: str | Path,
        hospital_dir: str | Path,
        tokenizer,
        num_hospitals: int = 3,
        max_prompt_length: int = 320,
        max_target_length: int = 64,
        limit: int = -1,
        hpo_embeddings_file: str | Path | None = None,
        hpo_ic_file: str | Path | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_hospitals = int(num_hospitals)
        self.max_prompt_length = int(max_prompt_length)
        self.max_target_length = int(max_target_length)
        rows = _read_json(data_file)
        if limit > 0:
            rows = rows[:limit]
        self.records = [self._normalize(row, idx) for idx, row in enumerate(rows)]
        self.hospitals = []
        for hospital_id in range(self.num_hospitals):
            path = Path(hospital_dir) / f"hospital_{hospital_id}.json"
            self.hospitals.append([self._normalize(row, idx) for idx, row in enumerate(_read_json(path))])
        if not hpo_embeddings_file or not hpo_ic_file:
            raise ValueError(
                "HPO cosine retrieval requires both hpo_embeddings_file and hpo_ic_file"
            )
        embeddings = _read_json_object(hpo_embeddings_file)
        ic_weights = _read_json_object(hpo_ic_file)
        self.retriever = HpoCosineRetriever(embeddings, ic_weights)
        self.examples = [self._build_example(idx) for idx in range(len(self.records))]

    def _normalize(self, row: dict[str, Any], idx: int) -> dict[str, Any]:
        return {
            "case_id": _case_id(row, idx),
            "hpo_codes": _hpo_codes(row),
            "phenotype_text": _phenotype_text(row),
            "disease_name": _disease_name(row),
            "omim_code": _omim_code(row),
        }

    def _retrieve(self, record: dict[str, Any], hospital_id: int) -> dict[str, Any]:
        candidates = self.hospitals[hospital_id]
        if not candidates:
            raise ValueError(f"Hospital {hospital_id} has no records")
        ranked = self.retriever.rank(record["hpo_codes"], candidates, top_k=2)
        if not ranked:
            raise ValueError(f"Hospital {hospital_id} has no candidates")
        selected = ranked[0].record
        if (
            str(selected.get("case_id") or selected.get("id") or "") == record["case_id"]
            and len(ranked) > 1
        ):
            selected = ranked[1].record
        return dict(selected)

    def _build_example(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        hospital_ids_all = []
        for hospital_id in range(self.num_hospitals):
            case = self._retrieve(record, hospital_id)
            prompt = build_hospital_prompt(
                hospital_id=hospital_id,
                case_disease=case["disease_name"],
                case_phenotype=case["phenotype_text"],
                query_phenotype=record["phenotype_text"],
            )
            text = _render_chat(
                self.tokenizer,
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
            hospital_ids_all.append(_tokenize(self.tokenizer, text, self.max_prompt_length))

        host_question = build_host_question(record["phenotype_text"])
        host_text = _render_chat(self.tokenizer, [{"role": "user", "content": host_question}], add_generation_prompt=True)
        target_text = build_target_answer(record["disease_name"])
        eos = self.tokenizer.eos_token or ""
        return {
            "case_id": record["case_id"],
            "omim_code": record["omim_code"],
            "hospital_ids_all": hospital_ids_all,
            "host_question_ids": _tokenize(self.tokenizer, host_text, self.max_prompt_length),
            "target_ids": _tokenize(self.tokenizer, target_text + eos, self.max_target_length)[: self.max_target_length],
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def _pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for row, seq in enumerate(seqs):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, : len(seq)] = 1
    return ids, mask


def collate_medlatent(batch: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    num_hospitals = len(batch[0]["hospital_ids_all"])
    hospital_ids_all = []
    hospital_mask_all = []
    for hospital_id in range(num_hospitals):
        ids, mask = _pad([item["hospital_ids_all"][hospital_id] for item in batch], pad_token_id)
        hospital_ids_all.append(ids)
        hospital_mask_all.append(mask)

    host_question_ids, host_question_mask = _pad([item["host_question_ids"] for item in batch], pad_token_id)
    target_ids, _ = _pad([item["target_ids"] for item in batch], pad_token_id)
    target_labels = torch.full_like(target_ids, IGNORE_INDEX)
    for idx, item in enumerate(batch):
        target_labels[idx, : len(item["target_ids"])] = torch.tensor(item["target_ids"], dtype=torch.long)
    return {
        "hospital_ids_all": hospital_ids_all,
        "hospital_mask_all": hospital_mask_all,
        "host_question_ids": host_question_ids,
        "host_question_mask": host_question_mask,
        "target_ids": target_ids,
        "target_labels": target_labels,
        "case_ids": [item["case_id"] for item in batch],
        "omim_codes": [item["omim_code"] for item in batch],
    }
