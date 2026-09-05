"""Canonical fixed undirected communication graph."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
import json
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DegreeStatistics:
    """Summary of graph node degrees."""

    minimum: int
    maximum: int
    mean: float


@dataclass(frozen=True, slots=True)
class CommunicationGraph:
    """An immutable simple undirected graph over stable integer agent IDs."""

    adjacency: NDArray[np.bool_]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    num_agents: int = field(init=False)
    agent_ids: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        raw_adjacency = np.asarray(self.adjacency)
        if raw_adjacency.ndim != 2 or raw_adjacency.shape[0] != raw_adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if raw_adjacency.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("adjacency must contain Boolean or binary numeric values")
        if not np.all((raw_adjacency == 0) | (raw_adjacency == 1)):
            raise ValueError("adjacency must contain only 0/1 values")

        adjacency = np.array(raw_adjacency, dtype=np.bool_, copy=True, order="C")
        if np.any(np.diag(adjacency)):
            raise ValueError("self-loops are not allowed")
        if not np.array_equal(adjacency, adjacency.T):
            raise ValueError("adjacency must be symmetric")

        immutable_adjacency = np.frombuffer(adjacency.tobytes(), dtype=np.bool_).reshape(adjacency.shape)
        object.__setattr__(self, "adjacency", immutable_adjacency)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(self, "num_agents", adjacency.shape[0])
        object.__setattr__(self, "agent_ids", tuple(range(adjacency.shape[0])))

    @property
    def graph_hash(self) -> str:
        """Return the stable SHA-256 identity of this graph and its metadata."""

        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def edge_list(self) -> tuple[tuple[int, int], ...]:
        """Return each undirected edge once in canonical agent-ID order."""

        return tuple(
            (source_id, target_id)
            for source_id in self.agent_ids
            for target_id in self.neighbors(source_id)
            if source_id < target_id
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible canonical graph representation."""

        return {
            "num_agents": self.num_agents,
            "edges": [list(edge) for edge in self.edge_list()],
            "metadata": _thaw_metadata(self.metadata),
        }

    def to_json(self) -> str:
        """Serialize the graph deterministically as compact JSON."""

        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    def save(self, path: str | Path) -> None:
        """Write the canonical graph JSON to ``path``."""

        Path(path).write_text(f"{self.to_json()}\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CommunicationGraph:
        """Construct a graph from its canonical edge-list representation."""

        if not isinstance(data, Mapping):
            raise ValueError("graph data must be a mapping")
        count = _validate_num_agents(data.get("num_agents"))
        edges = data.get("edges")
        if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
            raise ValueError("edges must be a sequence of agent-ID pairs")

        adjacency = np.zeros((count, count), dtype=np.bool_)
        seen_edges: set[tuple[int, int]] = set()
        for edge in edges:
            if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes)) or len(edge) != 2:
                raise ValueError("each edge must contain exactly two agent IDs")
            source_id = _validate_agent_id_for_count(edge[0], count)
            target_id = _validate_agent_id_for_count(edge[1], count)
            if source_id == target_id:
                raise ValueError("self-loops are not allowed")
            normalized_edge = tuple(sorted((source_id, target_id)))
            if normalized_edge in seen_edges:
                raise ValueError("duplicate undirected edge")
            seen_edges.add(normalized_edge)
            adjacency[source_id, target_id] = True
            adjacency[target_id, source_id] = True

        metadata = data.get("metadata", {})
        return cls(adjacency, metadata=metadata)

    @classmethod
    def from_json(cls, serialized: str) -> CommunicationGraph:
        """Construct a graph from canonical JSON."""

        try:
            data = json.loads(serialized)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("graph JSON is invalid") from error
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> CommunicationGraph:
        """Load a canonical graph JSON file."""

        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def neighbors(self, agent_id: int) -> tuple[int, ...]:
        """Return direct neighbors in ascending stable agent-ID order."""

        self._validate_agent_id(agent_id)
        return tuple(int(neighbor) for neighbor in np.flatnonzero(self.adjacency[agent_id]))

    def has_edge(self, sender_id: int, receiver_id: int) -> bool:
        """Return whether the two agents share a communication edge."""

        self._validate_agent_id(sender_id)
        self._validate_agent_id(receiver_id)
        return bool(self.adjacency[sender_id, receiver_id])

    def connected_components(self) -> tuple[tuple[int, ...], ...]:
        """Return components ordered by their smallest agent ID."""

        unseen = set(self.agent_ids)
        components: list[tuple[int, ...]] = []
        while unseen:
            start = min(unseen)
            component: list[int] = []
            queue = deque([start])
            unseen.remove(start)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in self.neighbors(current):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
            components.append(tuple(component))
        return tuple(components)

    def shortest_path_length(self, source_id: int, target_id: int) -> int | None:
        """Return the shortest path length, or ``None`` if no path exists."""

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

    def degree_statistics(self) -> DegreeStatistics:
        """Return minimum, maximum, and mean node degree."""

        degrees = np.count_nonzero(self.adjacency, axis=1)
        return DegreeStatistics(
            minimum=int(degrees.min()),
            maximum=int(degrees.max()),
            mean=float(degrees.mean()),
        )

    def diameter(self) -> int:
        """Return the diameter of a connected graph."""

        if len(self.connected_components()) != 1:
            raise ValueError("diameter is undefined for a disconnected graph")
        return max(
            self.shortest_path_length(source_id, target_id) or 0
            for source_id in self.agent_ids
            for target_id in self.agent_ids
        )

    def minimum_request_return_rounds(self, source_id: int, evidence_id: int) -> int | None:
        """Return the synchronous request-return lower bound of ``2d`` rounds."""

        distance = self.shortest_path_length(source_id, evidence_id)
        return None if distance is None else 2 * distance

    def _validate_agent_id(self, agent_id: Any) -> None:
        if isinstance(agent_id, bool) or not isinstance(agent_id, Integral):
            raise ValueError(f"invalid agent ID: {agent_id!r}")
        if agent_id < 0 or agent_id >= self.num_agents:
            raise ValueError(f"invalid agent ID: {agent_id!r}")


def complete_graph(num_agents: int) -> CommunicationGraph:
    """Build a complete simple graph."""

    count = _validate_num_agents(num_agents)
    adjacency = np.ones((count, count), dtype=np.bool_)
    np.fill_diagonal(adjacency, False)
    return CommunicationGraph(adjacency, metadata={"kind": "complete"})


def path_graph(num_agents: int) -> CommunicationGraph:
    """Build a path with agents ordered from 0 through ``num_agents - 1``."""

    count = _validate_num_agents(num_agents)
    adjacency = np.zeros((count, count), dtype=np.bool_)
    indices = np.arange(count - 1)
    adjacency[indices, indices + 1] = True
    adjacency[indices + 1, indices] = True
    return CommunicationGraph(adjacency, metadata={"kind": "path"})


def ring_graph(num_agents: int) -> CommunicationGraph:
    """Build a simple cycle graph."""

    count = _validate_num_agents(num_agents)
    if count < 3:
        raise ValueError("a ring graph requires at least three agents")
    adjacency = path_graph(count).adjacency.copy()
    adjacency[0, count - 1] = True
    adjacency[count - 1, 0] = True
    return CommunicationGraph(adjacency, metadata={"kind": "ring"})


def star_graph(num_agents: int) -> CommunicationGraph:
    """Build a star centered at agent 0."""

    count = _validate_num_agents(num_agents)
    adjacency = np.zeros((count, count), dtype=np.bool_)
    adjacency[0, 1:] = True
    adjacency[1:, 0] = True
    return CommunicationGraph(adjacency, metadata={"kind": "star", "center": 0})


def random_regular_graph(num_agents: int, degree: int, *, seed: int) -> CommunicationGraph:
    """Build a seeded simple undirected regular graph."""

    count = _validate_num_agents(num_agents)
    regular_degree = _validate_nonnegative_integer(degree, "degree")
    if regular_degree >= count:
        raise ValueError("degree must be smaller than num_agents")
    if count * regular_degree % 2:
        raise ValueError("num_agents * degree must be even")
    rng = np.random.default_rng(_validate_nonnegative_integer(seed, "seed"))

    adjacency = _circulant_regular_adjacency(count, regular_degree)
    permutation = rng.permutation(count)
    adjacency = adjacency[np.ix_(permutation, permutation)]
    _randomize_regular_edges(adjacency, rng)
    return CommunicationGraph(
        adjacency,
        metadata={"kind": "random_regular", "degree": regular_degree, "seed": int(seed)},
    )


def _validate_num_agents(num_agents: int) -> int:
    count = _validate_nonnegative_integer(num_agents, "num_agents")
    if count == 0:
        raise ValueError("num_agents must be positive")
    return count


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_agent_id_for_count(agent_id: Any, count: int) -> int:
    if isinstance(agent_id, bool) or not isinstance(agent_id, Integral):
        raise ValueError(f"invalid agent ID: {agent_id!r}")
    normalized_id = int(agent_id)
    if normalized_id < 0 or normalized_id >= count:
        raise ValueError(f"invalid agent ID: {agent_id!r}")
    return normalized_id


def _validate_probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _freeze_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    frozen: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        frozen[key] = _freeze_metadata_value(value)
    return MappingProxyType(frozen)


def _freeze_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("metadata floats must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_metadata(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_metadata_value(item) for item in value)
    raise ValueError(f"metadata value is not JSON-compatible: {value!r}")


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value


def _circulant_regular_adjacency(num_agents: int, degree: int) -> NDArray[np.bool_]:
    adjacency = np.zeros((num_agents, num_agents), dtype=np.bool_)
    agents = np.arange(num_agents)
    for offset in range(1, degree // 2 + 1):
        neighbors = (agents + offset) % num_agents
        adjacency[agents, neighbors] = True
        adjacency[neighbors, agents] = True
    if degree % 2:
        opposite = (agents + num_agents // 2) % num_agents
        adjacency[agents, opposite] = True
    return adjacency


def _randomize_regular_edges(adjacency: NDArray[np.bool_], rng: np.random.Generator) -> None:
    edges = [
        (source_id, target_id)
        for source_id in range(adjacency.shape[0])
        for target_id in range(source_id + 1, adjacency.shape[0])
        if adjacency[source_id, target_id]
    ]
    for _ in range(max(1, 4 * len(edges))):
        if len(edges) < 2:
            return
        first_index, second_index = rng.choice(len(edges), size=2, replace=False)
        first_edge = edges[first_index]
        second_edge = edges[second_index]
        candidate_edges = _edge_swap_candidates(first_edge, second_edge, rng)
        if candidate_edges is None:
            continue
        if any(adjacency[source_id, target_id] for source_id, target_id in candidate_edges):
            continue

        for source_id, target_id in (first_edge, second_edge):
            adjacency[source_id, target_id] = False
            adjacency[target_id, source_id] = False
        for source_id, target_id in candidate_edges:
            adjacency[source_id, target_id] = True
            adjacency[target_id, source_id] = True
        edges[first_index], edges[second_index] = candidate_edges


def _edge_swap_candidates(
    first_edge: tuple[int, int], second_edge: tuple[int, int], rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    first_source, first_target = first_edge
    second_source, second_target = second_edge
    if rng.integers(2):
        candidates = ((first_source, second_source), (first_target, second_target))
    else:
        candidates = ((first_source, second_target), (first_target, second_source))
    normalized = tuple(tuple(sorted(edge)) for edge in candidates)
    if normalized[0] == normalized[1] or any(source_id == target_id for source_id, target_id in normalized):
        return None
    return normalized
