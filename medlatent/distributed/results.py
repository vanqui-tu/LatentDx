"""Payload-free JSON result records for distributed episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .baselines import BaselineResult
from .config import DistributedBaselineConfig
from .fixtures import ComplementaryEvidenceFixture


@dataclass(frozen=True, slots=True)
class EpisodeLogRecord:
    config_hash: str
    graph_hash: str
    data_version: str
    model_id: str
    baseline: str
    episode_id: str
    source_id: int
    prediction: str | None
    target: str
    shortest_evidence_distance: int | None
    termination_reason: str
    contacted_agent_ids: tuple[int, ...]
    selected_edges_by_round: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    messages: int
    wire_bytes: int
    events: tuple[dict[str, Any], ...]

    @classmethod
    def from_baseline(
        cls,
        config: DistributedBaselineConfig,
        fixture: ComplementaryEvidenceFixture,
        result: BaselineResult,
        *,
        episode_id: str,
    ) -> EpisodeLogRecord:
        events = () if result.episode is None else result.episode.events
        sent = tuple(event for event in events if event.event_type == "sent")
        edges_by_round: dict[int, list[tuple[int, int]]] = {}
        for event in sent:
            assert event.receiver_id is not None
            edges_by_round.setdefault(event.round_index, []).append((event.agent_id, event.receiver_id))
        closest_distance = min(
            (
                fixture.graph.shortest_path_length(fixture.source_id, evidence_id)
                for evidence_id in fixture.necessary_agent_ids
            ),
            default=None,
        )
        return cls(
            config_hash=config.config_hash,
            graph_hash=fixture.graph.graph_hash,
            data_version=config.data_version,
            model_id=config.model_id,
            baseline=result.kind.value,
            episode_id=episode_id,
            source_id=fixture.source_id,
            prediction=result.prediction,
            target=fixture.target_label,
            shortest_evidence_distance=closest_distance,
            termination_reason="local_only" if result.episode is None else result.episode.termination_reason,
            contacted_agent_ids=result.contacted_agent_ids,
            selected_edges_by_round=tuple(
                (round_index, tuple(edges)) for round_index, edges in sorted(edges_by_round.items())
            ),
            messages=len(sent),
            wire_bytes=sum(event.wire_bytes for event in sent),
            events=tuple(_event_dict(event) for event in events),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "graph_hash": self.graph_hash,
            "data_version": self.data_version,
            "model_id": self.model_id,
            "baseline": self.baseline,
            "episode_id": self.episode_id,
            "source_id": self.source_id,
            "prediction": self.prediction,
            "target": self.target,
            "shortest_evidence_distance": self.shortest_evidence_distance,
            "termination_reason": self.termination_reason,
            "contacted_agent_ids": list(self.contacted_agent_ids),
            "selected_edges_by_round": [
                {"round": round_index, "edges": [list(edge) for edge in edges]}
                for round_index, edges in self.selected_edges_by_round
            ],
            "cost": {"messages": self.messages, "wire_bytes": self.wire_bytes},
            "events": list(self.events),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)


def _event_dict(event: Any) -> dict[str, Any]:
    return {
        "round": event.round_index,
        "type": event.event_type,
        "agent_id": event.agent_id,
        "message_id": event.message_id,
        "receiver_id": event.receiver_id,
        "wire_bytes": event.wire_bytes,
    }
