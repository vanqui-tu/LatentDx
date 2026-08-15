"""HPO-based local retrieval used by CROSSRARE-BENCH experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    record: Mapping[str, object]
    score: float


class HpoCosineRetriever:
    """Information-content-weighted HPO embedding retriever with cosine scoring."""

    def __init__(self, term_embeddings: Mapping[str, Sequence[float]], ic_weights: Mapping[str, float]):
        self.term_embeddings = {key: np.asarray(value, dtype=np.float32) for key, value in term_embeddings.items()}
        self.ic_weights = {key: float(value) for key, value in ic_weights.items()}
        if not self.term_embeddings:
            raise ValueError("term_embeddings must not be empty")
        self.embedding_dim = int(next(iter(self.term_embeddings.values())).shape[0])
        self._record_indexes: dict[int, tuple[Sequence[Mapping[str, object]], np.ndarray]] = {}

    def encode(self, hpo_codes: Sequence[str]) -> np.ndarray:
        vectors = []
        weights = []
        for code in hpo_codes:
            if code in self.term_embeddings and code in self.ic_weights:
                vectors.append(self.term_embeddings[code])
                weights.append(self.ic_weights[code])
        if not vectors:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        matrix = np.stack(vectors, axis=0)
        weight = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        denom = float(weight.sum())
        if denom <= 0.0:
            return matrix.mean(axis=0)
        return (matrix * weight).sum(axis=0) / denom

    @staticmethod
    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom == 0.0:
            return 0.0
        return float(np.dot(left, right) / denom)

    def rank(
        self,
        query_hpo_codes: Sequence[str],
        records: Sequence[Mapping[str, object]],
        *,
        top_k: int = 1,
    ) -> list[RetrievalResult]:
        query = self.encode(query_hpo_codes)
        cache_key = id(records)
        cached = self._record_indexes.get(cache_key)
        if cached is None or cached[0] is not records:
            candidate_vectors = np.stack(
                [self.encode(record.get("hpo_codes", [])) for record in records],  # type: ignore[arg-type]
                axis=0,
            ) if records else np.empty((0, self.embedding_dim), dtype=np.float32)
            norms = np.linalg.norm(candidate_vectors, axis=1)
            self._record_indexes[cache_key] = (records, candidate_vectors / np.maximum(norms[:, None], 1e-12))
            cached = self._record_indexes[cache_key]

        if not records or top_k <= 0:
            return []
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            scores = np.zeros(len(records), dtype=np.float32)
        else:
            scores = cached[1] @ (query / query_norm)
        count = min(int(top_k), len(records))
        indices = np.argsort(-scores, kind="stable")[:count]
        return [RetrievalResult(record=records[index], score=float(scores[index])) for index in indices]
