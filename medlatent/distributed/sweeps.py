"""Reproducible topology and communication-budget sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
from statistics import mean
from typing import Mapping, Sequence

from .agent import AgentRuntime
from .graph import CommunicationGraph, complete_graph, path_graph, random_regular_graph, ring_graph, star_graph
from .medical import HospitalPrivateStore, MedicalQueryRecord, sample_balanced_sources
from .medical_baselines import MedicalBaselineKind, run_medical_baseline


@dataclass(frozen=True, slots=True)
class SweepRecord:
    topology: str
    source_seed: int
    node_assignment_seed: int
    node_assignment_hash: str
    hospital_by_agent: tuple[tuple[int, int], ...]
    graph_seed: int
    routing_seed: int
    graph_hash: str
    graph_edges: tuple[tuple[int, int], ...]
    rounds: int
    max_fanout: int
    baseline: str
    channel: str
    episodes: int
    evidence_distance: int | None
    evidence_distance_stratum: str
    accuracy: float
    mean_messages: float
    mean_wire_bytes: float


def run_topology_budget_sweep(
    records: Sequence[MedicalQueryRecord],
    stores: Mapping[int, HospitalPrivateStore],
    *,
    num_agents: int,
    topology_kinds: Sequence[str] = ("complete", "ring", "star", "path"),
    source_seeds: Sequence[int] = (42,),
    node_assignment_seeds: Sequence[int] = (0,),
    graph_seeds: Sequence[int] = (42, 43, 44),
    routing_seeds: Sequence[int] = (0,),
    rounds: Sequence[int] = (1, 2, 4),
    fanouts: Sequence[int] = (1, 2),
    baselines: Sequence[MedicalBaselineKind] = tuple(MedicalBaselineKind),
    channel: str = "structured",
    random_regular_degree: int = 3,
) -> tuple[SweepRecord, ...]:
    if set(stores) != set(range(num_agents)):
        raise ValueError("stores must exactly match num_agents")
    _validate_seeds(source_seeds, "source_seeds")
    _validate_seeds(node_assignment_seeds, "node_assignment_seeds")
    _validate_seeds(graph_seeds, "graph_seeds")
    _validate_seeds(routing_seeds, "routing_seeds")
    if "random_regular" in topology_kinds and (num_agents < 6 or random_regular_degree < 3):
        raise ValueError("random_regular sweeps require at least six agents and degree at least three")
    output: list[SweepRecord] = []
    for node_assignment_seed in node_assignment_seeds:
        assigned_stores, hospital_by_agent = _assign_stores_to_nodes(stores, node_assignment_seed)
        node_assignment_hash = _assignment_hash(hospital_by_agent)
        for source_seed in source_seeds:
            episodes = sample_balanced_sources(records, split="sweep", num_agents=num_agents, seed=source_seed)
            for graph_seed in graph_seeds:
                for topology_kind in topology_kinds:
                    graph = _build_graph(topology_kind, num_agents, graph_seed, random_regular_degree)
                    strata = _episodes_by_evidence_distance(episodes, assigned_stores, graph)
                    for routing_seed in routing_seeds:
                        for round_count in rounds:
                            for fanout in fanouts:
                                for baseline in baselines:
                                    for (distance, stratum), stratum_episodes in strata.items():
                                        results = tuple(
                                            run_medical_baseline(
                                                episode,
                                                graph,
                                                {agent_id: AgentRuntime(agent_id, store) for agent_id, store in assigned_stores.items()},
                                                baseline,
                                                max_rounds=round_count,
                                                max_fanout=fanout,
                                                seed=routing_seed,
                                                channel=channel,
                                            )
                                            for episode in stratum_episodes
                                        )
                                        output.append(
                                            SweepRecord(
                                                topology=topology_kind,
                                                source_seed=source_seed,
                                                node_assignment_seed=node_assignment_seed,
                                                node_assignment_hash=node_assignment_hash,
                                                hospital_by_agent=hospital_by_agent,
                                                graph_seed=graph_seed,
                                                routing_seed=routing_seed,
                                                graph_hash=graph.graph_hash,
                                                graph_edges=graph.edge_list(),
                                                rounds=round_count,
                                                max_fanout=fanout,
                                                baseline=baseline.value,
                                                channel=channel,
                                                episodes=len(results),
                                                evidence_distance=distance,
                                                evidence_distance_stratum=stratum,
                                                accuracy=_accuracy(results, stratum_episodes),
                                                mean_messages=_mean_messages(results),
                                                mean_wire_bytes=mean(result.wire_bytes for result in results) if results else 0.0,
                                            )
                                        )
    return tuple(output)


def _episodes_by_evidence_distance(
    episodes: Sequence,
    stores: Mapping[int, HospitalPrivateStore],
    graph: CommunicationGraph,
) -> dict[tuple[int | None, str], list]:
    grouped: dict[tuple[int | None, str], list] = {}
    for episode in episodes:
        relevant = tuple(
            agent_id
            for agent_id, store in stores.items()
            if (ranked := store.retrieve(episode.query, limit=1))
            and ranked[0].label == episode.target_label
            and ranked[0].score > 0.0
        )
        reachable = tuple(
            distance
            for agent_id in relevant
            if (distance := graph.shortest_path_length(episode.source_id, agent_id)) is not None
        )
        if not relevant:
            key = (None, "no_proxy_evidence")
        elif not reachable:
            key = (None, "unreachable_proxy_evidence")
        else:
            distance = min(reachable)
            key = (distance, f"distance_{distance}")
        grouped.setdefault(key, []).append(episode)
    return grouped


def _validate_seeds(values: Sequence[int], name: str) -> None:
    if not values or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative integers")


def _assign_stores_to_nodes(
    stores: Mapping[int, HospitalPrivateStore], seed: int
) -> tuple[dict[int, HospitalPrivateStore], tuple[tuple[int, int], ...]]:
    hospital_ids = sorted(stores)
    if seed:
        random.Random(seed).shuffle(hospital_ids)
    assigned = {agent_id: stores[hospital_id] for agent_id, hospital_id in enumerate(hospital_ids)}
    return assigned, tuple((agent_id, store.agent_id) for agent_id, store in assigned.items())


def _assignment_hash(hospital_by_agent: tuple[tuple[int, int], ...]) -> str:
    encoded = json.dumps(hospital_by_agent, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _accuracy(results, episodes) -> float:
    return mean(result.prediction == episode.target_label for episode, result in zip(episodes, results)) if results else 0.0


def _mean_messages(results) -> float:
    return mean(sum(event.event_type == "sent" for event in result.episode.events) for result in results) if results else 0.0


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
