"""Strict configuration for synthetic distributed baseline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from numbers import Integral
from typing import Any, Mapping

from .graph import CommunicationGraph, complete_graph, path_graph, random_regular_graph, ring_graph, star_graph


@dataclass(frozen=True, slots=True)
class TopologyConfig:
    kind: str
    degree: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"complete", "path", "ring", "star", "random_regular"}:
            raise ValueError("unknown topology kind")
        if self.kind == "random_regular" and self.degree is None:
            raise ValueError("random_regular topology requires degree")
        if self.kind != "random_regular" and self.degree is not None:
            raise ValueError("degree is only valid for random_regular topology")
        if self.degree is not None:
            _nonnegative_integer(self.degree, "topology.degree")


@dataclass(frozen=True, slots=True)
class DistributedBaselineConfig:
    seed: int
    agents: int
    source_policy: str
    topology: TopologyConfig
    rounds: int
    max_fanout: int
    router: str
    channel: str
    aggregation: str
    source_id: int | None = None
    data_version: str = "synthetic-v1"
    model_id: str = "synthetic-structured-v1"
    deterministic: bool = True

    def __post_init__(self) -> None:
        _nonnegative_integer(self.seed, "seed")
        if _nonnegative_integer(self.agents, "agents") == 0:
            raise ValueError("agents must be positive")
        _nonnegative_integer(self.rounds, "rounds")
        _nonnegative_integer(self.max_fanout, "max_fanout")
        if self.source_policy not in {"fixed", "round_robin"}:
            raise ValueError("unknown source_policy")
        if self.router not in {"direct_neighbor", "flood_unvisited", "random_k", "heuristic_expertise"}:
            raise ValueError("unknown router")
        if self.channel != "structured":
            raise ValueError("unknown channel")
        if self.aggregation not in {"majority_vote", "mean_score", "max_confidence", "require_all_evidence"}:
            raise ValueError("unknown aggregation")
        if self.source_policy == "fixed" and self.source_id is None:
            raise ValueError("fixed source_policy requires source_id")
        if self.source_id is not None and not 0 <= _nonnegative_integer(self.source_id, "source_id") < self.agents:
            raise ValueError("source_id must be a valid agent ID")
        if not isinstance(self.data_version, str) or not self.data_version:
            raise ValueError("data_version must be a non-empty string")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be Boolean")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DistributedBaselineConfig:
        if not isinstance(data, Mapping):
            raise ValueError("config must be a mapping")
        allowed = {
            "seed",
            "agents",
            "source_policy",
            "topology",
            "rounds",
            "max_fanout",
            "router",
            "channel",
            "aggregation",
            "source_id",
            "data_version",
            "model_id",
            "deterministic",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown config settings: {sorted(unknown)}")
        required = allowed - {"source_id", "data_version", "model_id", "deterministic"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"missing config settings: {sorted(missing)}")
        topology_data = data["topology"]
        if not isinstance(topology_data, Mapping):
            raise ValueError("topology must be a mapping")
        topology_unknown = set(topology_data) - {"kind", "degree"}
        if topology_unknown:
            raise ValueError(f"unknown topology settings: {sorted(topology_unknown)}")
        if "kind" not in topology_data:
            raise ValueError("topology.kind is required")
        return cls(
            seed=data["seed"],
            agents=data["agents"],
            source_policy=data["source_policy"],
            topology=TopologyConfig(kind=topology_data["kind"], degree=topology_data.get("degree")),
            rounds=data["rounds"],
            max_fanout=data["max_fanout"],
            router=data["router"],
            channel=data["channel"],
            aggregation=data["aggregation"],
            source_id=data.get("source_id"),
            data_version=data.get("data_version", "synthetic-v1"),
            model_id=data.get("model_id", "synthetic-structured-v1"),
            deterministic=data.get("deterministic", True),
        )

    @property
    def config_hash(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    def build_graph(self) -> CommunicationGraph:
        builders = {
            "complete": complete_graph,
            "path": path_graph,
            "ring": ring_graph,
            "star": star_graph,
        }
        if self.topology.kind == "random_regular":
            return random_regular_graph(self.agents, self.topology.degree, seed=self.seed)
        return builders[self.topology.kind](self.agents)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
