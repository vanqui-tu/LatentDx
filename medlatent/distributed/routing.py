"""Fixed legal-neighbor routing baselines."""

from __future__ import annotations

from hashlib import sha256
from random import Random
from typing import Mapping, Protocol, Sequence

from .graph import CommunicationGraph


class Router(Protocol):
    def select_neighbors(
        self,
        graph: CommunicationGraph,
        sender_id: int,
        *,
        max_fanout: int,
        excluded_agent_ids: Sequence[int] = (),
        episode_id: str = "",
        round_index: int = 0,
    ) -> tuple[int, ...]:
        ...


class DirectNeighborRouter:
    """Select the first legal neighbors in stable agent-ID order."""

    def select_neighbors(
        self,
        graph: CommunicationGraph,
        sender_id: int,
        *,
        max_fanout: int,
        excluded_agent_ids: Sequence[int] = (),
        episode_id: str = "",
        round_index: int = 0,
    ) -> tuple[int, ...]:
        return _legal_neighbors(graph, sender_id, max_fanout, excluded_agent_ids)


class RandomKRouter:
    """Select a reproducible random subset of legal neighbors."""

    def __init__(self, *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed

    def select_neighbors(
        self,
        graph: CommunicationGraph,
        sender_id: int,
        *,
        max_fanout: int,
        excluded_agent_ids: Sequence[int] = (),
        episode_id: str = "",
        round_index: int = 0,
    ) -> tuple[int, ...]:
        candidates = _legal_neighbors(graph, sender_id, len(graph.neighbors(sender_id)), excluded_agent_ids)
        sample_size = min(_fanout(max_fanout), len(candidates))
        seed_material = f"{self.seed}:{episode_id}:{round_index}:{sender_id}".encode("utf-8")
        rng = Random(int.from_bytes(sha256(seed_material).digest()[:8], "big"))
        return tuple(sorted(rng.sample(candidates, sample_size)))


class PublicExpertiseRouter:
    """Select legal neighbors by overlap with public expertise terms."""

    def __init__(self, expertise: Mapping[int, Sequence[str]], query_terms: Sequence[str]) -> None:
        self.expertise = {int(agent_id): frozenset(str(term) for term in terms) for agent_id, terms in expertise.items()}
        self.query_terms = frozenset(str(term) for term in query_terms)

    def select_neighbors(
        self,
        graph: CommunicationGraph,
        sender_id: int,
        *,
        max_fanout: int,
        excluded_agent_ids: Sequence[int] = (),
        episode_id: str = "",
        round_index: int = 0,
    ) -> tuple[int, ...]:
        candidates = _legal_neighbors(graph, sender_id, len(graph.neighbors(sender_id)), excluded_agent_ids)
        return tuple(
            sorted(
                candidates,
                key=lambda neighbor: (-len(self.query_terms.intersection(self.expertise.get(neighbor, ()))), neighbor),
            )[: _fanout(max_fanout)]
        )


class FloodUnvisitedRouter(DirectNeighborRouter):
    """Forward only along legal edges outside the request ancestry."""


def _legal_neighbors(
    graph: CommunicationGraph,
    sender_id: int,
    max_fanout: int,
    excluded_agent_ids: Sequence[int],
) -> tuple[int, ...]:
    excluded = set(excluded_agent_ids)
    return tuple(neighbor for neighbor in graph.neighbors(sender_id) if neighbor not in excluded)[: _fanout(max_fanout)]


def _fanout(max_fanout: int) -> int:
    if isinstance(max_fanout, bool) or not isinstance(max_fanout, int) or max_fanout < 0:
        raise ValueError("max_fanout must be a non-negative integer")
    return max_fanout
