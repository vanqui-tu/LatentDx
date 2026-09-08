"""Fixed simple undirected communication graphs for M2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Sequence

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


def erdos_renyi_graph(num_agents: int, edge_probability: float, *, seed: int, connectivity: str = "allow_disconnected") -> CommunicationGraph:
    """Build a seeded Erdos-Renyi graph using optional NetworkX."""
    count = _num_agents(num_agents)
    probability = _probability(edge_probability, "edge_probability")
    _connectivity_policy(connectivity)
    network = _networkx().gnp_random_graph(count, probability, seed=_nonnegative(seed, "seed"))
    return _from_networkx(network, connectivity)


def watts_strogatz_graph(num_agents: int, nearest_neighbors: int, rewiring_probability: float, *, seed: int, connectivity: str = "allow_disconnected") -> CommunicationGraph:
    """Build a seeded Watts-Strogatz graph using optional NetworkX."""
    count = _num_agents(num_agents)
    neighbors = _nonnegative(nearest_neighbors, "nearest_neighbors")
    if neighbors >= count or neighbors % 2:
        raise ValueError("nearest_neighbors must be an even value smaller than num_agents")
    probability = _probability(rewiring_probability, "rewiring_probability")
    _connectivity_policy(connectivity)
    network = _networkx().watts_strogatz_graph(count, neighbors, probability, seed=_nonnegative(seed, "seed"))
    return _from_networkx(network, connectivity)


def stochastic_block_model_graph(block_sizes: Sequence[int], edge_probabilities: Sequence[Sequence[float]], *, seed: int, connectivity: str = "allow_disconnected") -> CommunicationGraph:
    """Build a seeded undirected stochastic block model graph."""
    if isinstance(block_sizes, (str, bytes)) or not block_sizes:
        raise ValueError("block_sizes must be a non-empty sequence")
    sizes = [_num_agents(size) for size in block_sizes]
    probabilities = np.asarray(edge_probabilities, dtype=float)
    block_count = len(sizes)
    if probabilities.shape != (block_count, block_count):
        raise ValueError("edge_probabilities must be square with one row per block")
    if not np.array_equal(probabilities, probabilities.T):
        raise ValueError("edge_probabilities must be symmetric")
    if not np.all((0.0 <= probabilities) & (probabilities <= 1.0)):
        raise ValueError("edge_probabilities must be between 0 and 1")
    _connectivity_policy(connectivity)
    network = _networkx().stochastic_block_model(sizes, probabilities.tolist(), seed=_nonnegative(seed, "seed"))
    return _from_networkx(network, connectivity)


def _networkx() -> Any:
    try:
        import networkx as nx
    except ImportError as error:
        raise RuntimeError("networkx is required for optional graph generators; install medlatent[graphs]") from error
    return nx


def _from_networkx(network: Any, connectivity: str) -> CommunicationGraph:
    graph = CommunicationGraph(_networkx().to_numpy_array(network, nodelist=range(len(network)), dtype=np.bool_))
    if connectivity == "connected" and not _is_connected(graph):
        raise ValueError("generated graph is disconnected under connectivity='connected'")
    return graph


def _is_connected(graph: CommunicationGraph) -> bool:
    reached, frontier = {0}, [0]
    while frontier:
        for neighbor in graph.neighbors(frontier.pop()):
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return len(reached) == graph.num_agents


def _connectivity_policy(value: str) -> None:
    if value not in {"connected", "allow_disconnected"}:
        raise ValueError("connectivity must be 'connected' or 'allow_disconnected'")


def _nonnegative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _num_agents(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("num_agents must be a positive integer")
    return int(value)
