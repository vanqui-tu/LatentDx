"""Typed transport envelopes and structured baseline payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from math import isfinite
from numbers import Integral
from typing import Any, Mapping, Sequence
from uuid import uuid4


class MessageKind(str, Enum):
    REQUEST = "REQUEST"
    EVIDENCE = "EVIDENCE"
    PROPOSAL = "PROPOSAL"
    ACK = "ACK"


@dataclass(frozen=True, slots=True)
class RequestPayload:
    request_type: str
    candidate_labels: tuple[str, ...] = ()
    visited_agent_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_type:
            raise ValueError("request_type must not be empty")
        object.__setattr__(self, "candidate_labels", _labels(self.candidate_labels))
        object.__setattr__(self, "visited_agent_ids", _agent_ids(self.visited_agent_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "request",
            "request_type": self.request_type,
            "candidate_labels": list(self.candidate_labels),
            "visited_agent_ids": list(self.visited_agent_ids),
        }


@dataclass(frozen=True, slots=True)
class EvidencePayload:
    candidate_label: str
    evidence_present: bool
    score: float

    def __post_init__(self) -> None:
        _label(self.candidate_label, "candidate_label")
        _finite(self.score, "score")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "evidence",
            "candidate_label": self.candidate_label,
            "evidence_present": self.evidence_present,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class ProposalPayload:
    candidate_label: str
    score: float
    confidence: float

    def __post_init__(self) -> None:
        _label(self.candidate_label, "candidate_label")
        _finite(self.score, "score")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "proposal",
            "candidate_label": self.candidate_label,
            "score": self.score,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class AckPayload:
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ack", "accepted": self.accepted}


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
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must not be empty")
        _label(self.candidate_label, "candidate_label")
        _finite(self.score, "score")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "text_proposal",
            "text": self.text,
            "candidate_label": self.candidate_label,
            "score": self.score,
            "confidence": self.confidence,
        }


StructuredPayload = RequestPayload | EvidencePayload | ProposalPayload | AckPayload
TextPayload = TextRequestPayload | TextProposalPayload
Payload = StructuredPayload | TextPayload


@dataclass(frozen=True, slots=True)
class CommunicationCost:
    logical_size: int
    wire_bytes: int
    latent_positions: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("logical_size", self.logical_size),
            ("wire_bytes", self.wire_bytes),
            ("latent_positions", self.latent_positions),
        ):
            _nonnegative_integer(value, name)

    @classmethod
    def from_payload(cls, payload: Payload) -> CommunicationCost:
        wire_bytes = len(_canonical_json(payload.to_dict()).encode("utf-8"))
        return cls(logical_size=wire_bytes, wire_bytes=wire_bytes)

    def to_dict(self) -> dict[str, int]:
        return {
            "logical_size": self.logical_size,
            "wire_bytes": self.wire_bytes,
            "latent_positions": self.latent_positions,
        }


@dataclass(frozen=True, slots=True)
class MessageEnvelope:
    message_id: str
    episode_id: str
    round_sent: int
    sender_id: int
    receiver_id: int
    kind: MessageKind
    ttl: int
    payload: Payload
    cost: CommunicationCost
    parent_message_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (("message_id", self.message_id), ("episode_id", self.episode_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.parent_message_id is not None and (
            not isinstance(self.parent_message_id, str) or not self.parent_message_id
        ):
            raise ValueError("parent_message_id must be a non-empty string or None")
        for name, value in (
            ("round_sent", self.round_sent),
            ("sender_id", self.sender_id),
            ("receiver_id", self.receiver_id),
            ("ttl", self.ttl),
        ):
            _nonnegative_integer(value, name)
        if self.sender_id == self.receiver_id:
            raise ValueError("messages cannot target their sender")
        if not isinstance(self.kind, MessageKind):
            raise ValueError("kind must be a MessageKind")
        if not _kind_matches_payload(self.kind, self.payload):
            raise ValueError("message kind does not match payload type")

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        round_sent: int,
        sender_id: int,
        receiver_id: int,
        kind: MessageKind,
        ttl: int,
        payload: Payload,
        parent_message_id: str | None = None,
        message_id: str | None = None,
        cost: CommunicationCost | None = None,
    ) -> MessageEnvelope:
        return cls(
            message_id=message_id or new_message_id(),
            episode_id=episode_id,
            round_sent=round_sent,
            sender_id=sender_id,
            receiver_id=receiver_id,
            kind=kind,
            ttl=ttl,
            payload=payload,
            cost=CommunicationCost.from_payload(payload) if cost is None else cost,
            parent_message_id=parent_message_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "episode_id": self.episode_id,
            "round_sent": self.round_sent,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "kind": self.kind.value,
            "ttl": self.ttl,
            "payload": self.payload.to_dict(),
            "cost": self.cost.to_dict(),
            "parent_message_id": self.parent_message_id,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MessageEnvelope:
        if not isinstance(data, Mapping):
            raise ValueError("message data must be a mapping")
        try:
            cost_data = data["cost"]
            cost = CommunicationCost(
                logical_size=cost_data["logical_size"],
                wire_bytes=cost_data["wire_bytes"],
                latent_positions=cost_data.get("latent_positions", 0),
            )
            return cls(
                message_id=data["message_id"],
                episode_id=data["episode_id"],
                round_sent=data["round_sent"],
                sender_id=data["sender_id"],
                receiver_id=data["receiver_id"],
                kind=MessageKind(data["kind"]),
                ttl=data["ttl"],
                payload=_payload_from_dict(data["payload"]),
                cost=cost,
                parent_message_id=data.get("parent_message_id"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("message data is invalid") from error

    @classmethod
    def from_json(cls, serialized: str) -> MessageEnvelope:
        try:
            data = json.loads(serialized)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("message JSON is invalid") from error
        return cls.from_dict(data)


def new_message_id() -> str:
    return uuid4().hex


def _payload_from_dict(data: Mapping[str, Any]) -> Payload:
    if not isinstance(data, Mapping):
        raise ValueError("payload must be a mapping")
    payload_type = data.get("type")
    try:
        if payload_type == "request":
            return RequestPayload(
                data["request_type"],
                tuple(data.get("candidate_labels", ())),
                tuple(data.get("visited_agent_ids", ())),
            )
        if payload_type == "evidence":
            return EvidencePayload(data["candidate_label"], data["evidence_present"], data["score"])
        if payload_type == "proposal":
            return ProposalPayload(data["candidate_label"], data["score"], data["confidence"])
        if payload_type == "ack":
            return AckPayload(data["accepted"])
        if payload_type == "text_request":
            return TextRequestPayload(data["text"], tuple(data.get("visited_agent_ids", ())))
        if payload_type == "text_proposal":
            return TextProposalPayload(data["text"], data["candidate_label"], data["score"], data["confidence"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("payload data is invalid") from error
    raise ValueError("unknown payload type")


def _kind_matches_payload(kind: MessageKind, payload: Payload) -> bool:
    return {
        MessageKind.REQUEST: RequestPayload,
        MessageKind.EVIDENCE: EvidencePayload,
        MessageKind.PROPOSAL: ProposalPayload,
        MessageKind.ACK: AckPayload,
    }[kind] is type(payload) or (
        kind is MessageKind.REQUEST and isinstance(payload, TextRequestPayload)
    ) or (kind is MessageKind.PROPOSAL and isinstance(payload, TextProposalPayload))


def _canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _labels(labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)):
        raise ValueError("candidate_labels must be a sequence of labels")
    return tuple(_label(label, "candidate label") for label in labels)


def _agent_ids(agent_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(agent_ids, (str, bytes)):
        raise ValueError("visited_agent_ids must be a sequence of agent IDs")
    normalized = tuple(_nonnegative_integer(agent_id, "visited agent ID") for agent_id in agent_ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError("visited_agent_ids must not contain duplicates")
    return normalized


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
