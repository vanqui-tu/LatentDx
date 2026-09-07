"""In-memory envelopes and bounded M2 payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import isfinite
from numbers import Integral
from typing import Any, Sequence
from uuid import uuid4


class MessageKind(str, Enum):
    REQUEST = "REQUEST"
    PROPOSAL = "PROPOSAL"


@dataclass(frozen=True, slots=True)
class RequestPayload:
    visited_agent_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "visited_agent_ids", _agent_ids(self.visited_agent_ids))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "request", "visited_agent_ids": list(self.visited_agent_ids)}


@dataclass(frozen=True, slots=True)
class ProposalPayload:
    candidate_label: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_label, str) or not self.candidate_label:
            raise ValueError("candidate_label must be a non-empty string")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not isfinite(self.score):
            raise ValueError("score must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "proposal", "candidate_label": self.candidate_label, "score": self.score}


@dataclass(frozen=True, slots=True)
class TextRequestPayload:
    text: str
    visited_agent_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must not be empty")
        object.__setattr__(self, "visited_agent_ids", _agent_ids(self.visited_agent_ids))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text_request", "text": self.text, "visited_agent_ids": list(self.visited_agent_ids)}


@dataclass(frozen=True, slots=True)
class TextProposalPayload:
    text: str
    candidate_label: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must not be empty")
        ProposalPayload(self.candidate_label, self.score)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text_proposal", "text": self.text, "candidate_label": self.candidate_label, "score": self.score}


Payload = RequestPayload | ProposalPayload | TextRequestPayload | TextProposalPayload
Request = RequestPayload | TextRequestPayload
Proposal = ProposalPayload | TextProposalPayload


@dataclass(frozen=True, slots=True)
class MessageEnvelope:
    message_id: str
    episode_id: str
    round_sent: int
    sender_id: int
    receiver_id: int
    kind: MessageKind
    payload: Payload
    wire_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        for name in ("message_id", "episode_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("round_sent", "sender_id", "receiver_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.sender_id == self.receiver_id:
            raise ValueError("messages cannot target their sender")
        if not _matches(self.kind, self.payload):
            raise ValueError("message kind does not match payload type")
        object.__setattr__(self, "wire_bytes", len(_json(self.payload.to_dict()).encode("utf-8")))

    @classmethod
    def create(
        cls, *, episode_id: str, round_sent: int, sender_id: int, receiver_id: int,
        kind: MessageKind, payload: Payload, message_id: str | None = None,
    ) -> MessageEnvelope:
        return cls(message_id or uuid4().hex, episode_id, round_sent, sender_id, receiver_id, kind, payload)


def _matches(kind: MessageKind, payload: Payload) -> bool:
    return (kind is MessageKind.REQUEST and isinstance(payload, (RequestPayload, TextRequestPayload))) or (
        kind is MessageKind.PROPOSAL and isinstance(payload, (ProposalPayload, TextProposalPayload))
    )


def _agent_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("visited_agent_ids must be a sequence of agent IDs")
    normalized = tuple(int(value) for value in values)
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in values):
        raise ValueError("visited agent ID must be a non-negative integer")
    if len(set(normalized)) != len(normalized):
        raise ValueError("visited_agent_ids must not contain duplicates")
    return normalized


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
