"""Fixed simple undirected communication graphs for M2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CommunicationGraph:
    """An immutable simple undirected graph over integer agent IDs."""

    adjacency: NDArray[np.bool_]
    num_agents: int = field(init=False)
    agent_ids: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        raw = np.asarray(self.adjacency)
        if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if raw.dtype.kind not in {"b", "i", "u", "f"} or not np.all((raw == 0) | (raw == 1)):
            raise ValueError("adjacency must contain only 0/1 values")
        adjacency = np.array(raw, dtype=np.bool_, copy=True, order="C")
        if np.any(np.diag(adjacency)):
            raise ValueError("self-loops are not allowed")
        if not np.array_equal(adjacency, adjacency.T):
            raise ValueError("adjacency must be symmetric")
        immutable = np.frombuffer(adjacency.tobytes(), dtype=np.bool_).reshape(adjacency.shape)
        object.__setattr__(self, "adjacency", immutable)
        object.__setattr__(self, "num_agents", adjacency.shape[0])
        object.__setattr__(self, "agent_ids", tuple(range(adjacency.shape[0])))

    def neighbors(self, agent_id: int) -> tuple[int, ...]:
        self._validate_agent_id(agent_id)
        return tuple(int(value) for value in np.flatnonzero(self.adjacency[agent_id]))

    def has_edge(self, sender_id: int, receiver_id: int) -> bool:
        self._validate_agent_id(sender_id)
        self._validate_agent_id(receiver_id)
        return bool(self.adjacency[sender_id, receiver_id])

    def shortest_path_length(self, source_id: int, target_id: int) -> int | None:
        self._validate_agent_id(source_id)
        self._validate_agent_id(target_id)
        if source_id == target_id:
            return 0
        queue = deque([(source_id, 0)])
        visited = {source_id}
        while queue:
            current, distance = queue.popleft()
            for neighbor in self.neighbors(current):
                if neighbor == target_id:
                    return distance + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return None

    def edge_list(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (source_id, target_id)
            for source_id in self.agent_ids
            for target_id in self.neighbors(source_id)
            if source_id < target_id
        )

    def _validate_agent_id(self, agent_id: Any) -> None:
        if isinstance(agent_id, bool) or not isinstance(agent_id, Integral) or agent_id not in self.agent_ids:
            raise ValueError(f"invalid agent ID: {agent_id!r}")


def complete_graph(num_agents: int) -> CommunicationGraph:
    count = _num_agents(num_agents)
    adjacency = np.ones((count, count), dtype=np.bool_)
    np.fill_diagonal(adjacency, False)
    return CommunicationGraph(adjacency)


def path_graph(num_agents: int) -> CommunicationGraph:
    count = _num_agents(num_agents)
    adjacency = np.zeros((count, count), dtype=np.bool_)
    indices = np.arange(count - 1)
    adjacency[indices, indices + 1] = True
    adjacency[indices + 1, indices] = True
    return CommunicationGraph(adjacency)


def ring_graph(num_agents: int) -> CommunicationGraph:
    count = _num_agents(num_agents)
    if count < 3:
        raise ValueError("a ring graph requires at least three agents")
    adjacency = path_graph(count).adjacency.copy()
    adjacency[0, -1] = True
    adjacency[-1, 0] = True
    return CommunicationGraph(adjacency)


def _num_agents(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("num_agents must be a positive integer")
    return int(value)
