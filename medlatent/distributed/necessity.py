"""Counterfactual necessity audit for distributed medical episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .agent import AgentRuntime
from .graph import CommunicationGraph
from .medical import EmptyPrivateStore, HospitalPrivateStore, MedicalEpisode
from .medical_baselines import MedicalBaselineKind, MedicalBaselineResult, run_medical_baseline


@dataclass(frozen=True, slots=True)
class NecessityEpisodeAudit:
    case_id: str
    source_id: int
    target: str
    proxy_relevant_agent_ids: tuple[int, ...]
    full_prediction: str | None
    full_correct: bool
    removal_predictions: tuple[tuple[int, str | None], ...]

    @property
    def collaboration_required(self) -> bool:
        return self.full_correct and any(prediction != self.target for _, prediction in self.removal_predictions)


@dataclass(frozen=True, slots=True)
class NecessityAuditReport:
    baseline: str
    episodes: tuple[NecessityEpisodeAudit, ...]

    @property
    def accuracy(self) -> float:
        return _mean(episode.full_correct for episode in self.episodes)

    @property
    def collaboration_required_rate(self) -> float:
        return _mean(episode.collaboration_required for episode in self.episodes)

    @property
    def proxy_recall(self) -> float:
        relevant = [episode for episode in self.episodes if episode.proxy_relevant_agent_ids]
        recalls = []
        for episode in relevant:
            affected = {
                agent_id for agent_id, prediction in episode.removal_predictions if prediction != episode.full_prediction
            }
            recalls.append(bool(set(episode.proxy_relevant_agent_ids).intersection(affected)))
        return _mean(recalls)


def audit_medical_necessity(
    episodes: Sequence[MedicalEpisode],
    graph: CommunicationGraph,
    stores: Mapping[int, HospitalPrivateStore],
    *,
    baseline: MedicalBaselineKind,
    max_rounds: int,
    max_fanout: int,
    score_threshold: float = 0.0,
    seed: int = 0,
    channel: str = "structured",
) -> NecessityAuditReport:
    if set(stores) != set(graph.agent_ids):
        raise ValueError("stores must exactly match graph IDs")
    if score_threshold < 0.0:
        raise ValueError("score_threshold must be non-negative")
    audits: list[NecessityEpisodeAudit] = []
    for episode in episodes:
        relevant = tuple(
            agent_id
            for agent_id, store in stores.items()
            if _is_proxy_relevant(store, episode, score_threshold)
        )
        full_result = _run(episode, graph, stores, baseline, max_rounds, max_fanout, seed, channel)
        removals = tuple(
            (agent_id, _run_without(episode, graph, stores, agent_id, baseline, max_rounds, max_fanout, seed, channel).prediction)
            for agent_id in graph.agent_ids
        )
        audits.append(
            NecessityEpisodeAudit(
                case_id=episode.query.case_id,
                source_id=episode.source_id,
                target=episode.target_label,
                proxy_relevant_agent_ids=relevant,
                full_prediction=full_result.prediction,
                full_correct=full_result.prediction == episode.target_label,
                removal_predictions=removals,
            )
        )
    return NecessityAuditReport(baseline.value, tuple(audits))


def _is_proxy_relevant(store: HospitalPrivateStore, episode: MedicalEpisode, threshold: float) -> bool:
    ranked = store.retrieve(episode.query, limit=1)
    return bool(ranked and ranked[0].label == episode.target_label and ranked[0].score > threshold)


def _run(episode, graph, stores, baseline, rounds, fanout, seed, channel) -> MedicalBaselineResult:
    agents = {agent_id: AgentRuntime(agent_id, store) for agent_id, store in stores.items()}
    return run_medical_baseline(episode, graph, agents, baseline, max_rounds=rounds, max_fanout=fanout, seed=seed, channel=channel)


def _run_without(episode, graph, stores, removed_id, baseline, rounds, fanout, seed, channel) -> MedicalBaselineResult:
    counterfactual = {
        agent_id: EmptyPrivateStore() if agent_id == removed_id else store
        for agent_id, store in stores.items()
    }
    return _run(episode, graph, counterfactual, baseline, rounds, fanout, seed, channel)


def _mean(values) -> float:
    values = tuple(bool(value) for value in values)
    return sum(values) / len(values) if values else 0.0
