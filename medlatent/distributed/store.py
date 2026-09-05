"""Private local knowledge-store contracts for distributed episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar, runtime_checkable


RecordT = TypeVar("RecordT", covariant=True)


@runtime_checkable
class PrivateKnowledgeStore(Protocol[RecordT]):
    """A store retrievable only by its owning agent runtime."""

    def retrieve(self, query: object, *, limit: int | None = None) -> tuple[RecordT, ...]:
        ...


@dataclass(frozen=True, slots=True)
class SyntheticKnowledgeRecord:
    record_id: str
    terms: tuple[str, ...]
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        if not self.record_id or not self.label:
            raise ValueError("record_id and label must not be empty")
        object.__setattr__(self, "terms", tuple(self.terms))


class SyntheticKnowledgeStore:
    """Small in-memory store for protocol tests and CPU simulations."""

    __slots__ = ("__records",)

    def __init__(self, records: Sequence[SyntheticKnowledgeRecord]) -> None:
        self.__records = tuple(records)

    def retrieve(self, query: object, *, limit: int | None = None) -> tuple[SyntheticKnowledgeRecord, ...]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        query_terms = _query_terms(query)
        matches = tuple(record for record in self.__records if query_terms.intersection(record.terms))
        if limit is None:
            return matches
        return matches[:limit]


def _query_terms(query: object) -> set[str]:
    if isinstance(query, str):
        return {query}
    if isinstance(query, Sequence) and not isinstance(query, (str, bytes)):
        return {term for term in query if isinstance(term, str)}
    return set()
