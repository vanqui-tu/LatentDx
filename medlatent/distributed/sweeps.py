"""Reproducible topology and communication-budget sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence

from .agent import AgentRuntime
from .graph import CommunicationGraph, complete_graph, path_graph, random_regular_graph, ring_graph, star_graph
from .medical import HospitalPrivateStore, MedicalQueryRecord, sample_balanced_sources
from .medical_baselines import MedicalBaselineKind, run_medical_baseline


@dataclass(frozen=True, slots=True)
class SweepRecord:
    topology: str
    graph_seed: int
    graph_hash: str
    rounds: int
    max_fanout: int
    baseline: str
    channel: str
    episodes: int
    accuracy: float
    mean_messages: float
    mean_wire_bytes: float
    mean_contact_distance: float


def run_topology_budget_sweep(
    records: Sequence[MedicalQueryRecord],
    stores: Mapping[int, HospitalPrivateStore],
    *,
    num_agents: int,
    topology_kinds: Sequence[str] = ("complete", "ring", "star", "random_regular"),
    graph_seeds: Sequence[int] = (42, 43, 44),
    rounds: Sequence[int] = (1, 2, 4),
    fanouts: Sequence[int] = (1, 2),
    baselines: Sequence[MedicalBaselineKind] = tuple(MedicalBaselineKind),
    channel: str = "structured",
    random_regular_degree: int = 2,
) -> tuple[SweepRecord, ...]:
    if set(stores) != set(range(num_agents)):
        raise ValueError("stores must exactly match num_agents")
    output: list[SweepRecord] = []
    for graph_seed in graph_seeds:
        episodes = sample_balanced_sources(records, split="sweep", num_agents=num_agents, seed=graph_seed)
        for topology_kind in topology_kinds:
            graph = _build_graph(topology_kind, num_agents, graph_seed, random_regular_degree)
            for round_count in rounds:
                for fanout in fanouts:
                    for baseline in baselines:
                        results = tuple(
                            run_medical_baseline(
                                episode,
                                graph,
                                {agent_id: AgentRuntime(agent_id, store) for agent_id, store in stores.items()},
                                baseline,
                                max_rounds=round_count,
                                max_fanout=fanout,
                                seed=graph_seed,
                                channel=channel,
                            )
                            for episode in episodes
                        )
                        distances = [
                            graph.shortest_path_length(episode.source_id, agent_id)
                            for episode, result in zip(episodes, results)
                            for agent_id in result.contacted_agent_ids
                            if agent_id is not None and graph.shortest_path_length(episode.source_id, agent_id) is not None
                        ]
                        output.append(
                            SweepRecord(
                                topology=topology_kind,
                                graph_seed=graph_seed,
                                graph_hash=graph.graph_hash,
                                rounds=round_count,
                                max_fanout=fanout,
                                baseline=baseline.value,
                                channel=channel,
                                episodes=len(results),
                                accuracy=mean(result.prediction == episode.target_label for episode, result in zip(episodes, results)) if results else 0.0,
                                mean_messages=mean(sum(event.event_type == "sent" for event in result.episode.events) for result in results) if results else 0.0,
                                mean_wire_bytes=mean(result.wire_bytes for result in results) if results else 0.0,
                                mean_contact_distance=mean(distances) if distances else 0.0,
                            )
                        )
    return tuple(output)


def _build_graph(kind: str, num_agents: int, seed: int, degree: int) -> CommunicationGraph:
    if kind == "complete":
        return complete_graph(num_agents)
    if kind == "path":
        return path_graph(num_agents)
    if kind == "ring":
        return ring_graph(num_agents)
    if kind == "star":
        return star_graph(num_agents)
    if kind == "random_regular":
        return random_regular_graph(num_agents, degree, seed=seed)
    raise ValueError(f"unknown topology kind: {kind}")
