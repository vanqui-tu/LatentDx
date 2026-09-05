"""Deterministic structured proposal aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .messages import ProposalPayload


@dataclass(frozen=True, slots=True)
class AggregatedProposal:
    candidate_label: str
    score: float
    confidence: float
    support: int


def majority_vote(proposals: Iterable[ProposalPayload]) -> AggregatedProposal | None:
    grouped = _group(proposals)
    if not grouped:
        return None
    label, values = max(grouped.items(), key=lambda item: (len(item[1]), _mean_score(item[1]), item[0]))
    return AggregatedProposal(label, _mean_score(values), _mean_confidence(values), len(values))


def mean_score(proposals: Iterable[ProposalPayload]) -> AggregatedProposal | None:
    grouped = _group(proposals)
    if not grouped:
        return None
    label, values = max(grouped.items(), key=lambda item: (_mean_score(item[1]), _mean_confidence(item[1]), item[0]))
    return AggregatedProposal(label, _mean_score(values), _mean_confidence(values), len(values))


def max_confidence(proposals: Iterable[ProposalPayload]) -> AggregatedProposal | None:
    values = tuple(proposals)
    if not values:
        return None
    selected = max(values, key=lambda proposal: (proposal.confidence, proposal.score, proposal.candidate_label))
    return AggregatedProposal(selected.candidate_label, selected.score, selected.confidence, 1)


def require_all_evidence(
    proposals: Iterable[ProposalPayload],
    *,
    target_label: str,
    required_evidence_labels: Sequence[str],
) -> AggregatedProposal | None:
    values = tuple(proposals)
    observed = {proposal.candidate_label for proposal in values}
    if not set(required_evidence_labels).issubset(observed):
        return None
    required = tuple(proposal for proposal in values if proposal.candidate_label in required_evidence_labels)
    return AggregatedProposal(target_label, _mean_score(required), _mean_confidence(required), len(required))


def _group(proposals: Iterable[ProposalPayload]) -> dict[str, list[ProposalPayload]]:
    grouped: dict[str, list[ProposalPayload]] = defaultdict(list)
    for proposal in proposals:
        grouped[proposal.candidate_label].append(proposal)
    return grouped


def _mean_score(proposals: Sequence[ProposalPayload]) -> float:
    return sum(proposal.score for proposal in proposals) / len(proposals)


def _mean_confidence(proposals: Sequence[ProposalPayload]) -> float:
    return sum(proposal.confidence for proposal in proposals) / len(proposals)
