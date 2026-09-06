"""Bounded structured message representation."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .messages import (
    CommunicationCost,
    StructuredPayload,
    TextPayload,
    TextProposalPayload,
    TextRequestPayload,
    _payload_from_dict,
)


class StructuredChannel:
    """JSON structured payload transport with a strict byte limit."""

    def __init__(self, *, max_wire_bytes: int = 4_096) -> None:
        if isinstance(max_wire_bytes, bool) or not isinstance(max_wire_bytes, int) or max_wire_bytes <= 0:
            raise ValueError("max_wire_bytes must be a positive integer")
        self.max_wire_bytes = max_wire_bytes

    def encode(self, payload: StructuredPayload) -> bytes:
        serialized = json.dumps(payload.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(serialized) > self.max_wire_bytes:
            raise ValueError("structured payload exceeds max_wire_bytes")
        return serialized

    def decode(self, encoded: bytes) -> StructuredPayload:
        if not isinstance(encoded, bytes):
            raise TypeError("encoded payload must be bytes")
        if len(encoded) > self.max_wire_bytes:
            raise ValueError("structured payload exceeds max_wire_bytes")
        try:
            data = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("structured payload is invalid") from error
        return _payload_from_dict(data)

    def cost(self, payload: StructuredPayload) -> CommunicationCost:
        encoded = self.encode(payload)
        return CommunicationCost(logical_size=len(encoded), wire_bytes=len(encoded))


class TextChannel:
    """Bounded deterministic UTF-8 text transport with token accounting."""

    def __init__(self, *, max_tokens: int = 128, max_wire_bytes: int = 4_096) -> None:
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(max_wire_bytes, int) or max_wire_bytes <= 0:
            raise ValueError("max_wire_bytes must be a positive integer")
        self.max_tokens = max_tokens
        self.max_wire_bytes = max_wire_bytes

    def encode(self, payload: TextPayload) -> bytes:
        if not isinstance(payload, (TextRequestPayload, TextProposalPayload)):
            raise TypeError("TextChannel accepts only text payloads")
        tokens = _text_tokens(payload.text)
        if tokens > self.max_tokens:
            raise ValueError("text payload exceeds max_tokens")
        encoded = json.dumps(payload.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(encoded) > self.max_wire_bytes:
            raise ValueError("text payload exceeds max_wire_bytes")
        return encoded

    def decode(self, encoded: bytes) -> TextPayload:
        if not isinstance(encoded, bytes):
            raise TypeError("encoded payload must be bytes")
        if len(encoded) > self.max_wire_bytes:
            raise ValueError("text payload exceeds max_wire_bytes")
        try:
            payload = _payload_from_dict(json.loads(encoded.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("text payload is invalid") from error
        if not isinstance(payload, (TextRequestPayload, TextProposalPayload)):
            raise ValueError("text payload has a non-text type")
        if _text_tokens(payload.text) > self.max_tokens:
            raise ValueError("text payload exceeds max_tokens")
        return payload

    def cost(self, payload: TextPayload) -> CommunicationCost:
        encoded = self.encode(payload)
        return CommunicationCost(logical_size=_text_tokens(payload.text), wire_bytes=len(encoded))


def _text_tokens(text: str) -> int:
    return len(text.split())


def find_raw_substring_leaks(text: str, private_values: Sequence[str]) -> tuple[str, ...]:
    """Return private values exposed verbatim in a text message."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(dict.fromkeys(value for value in private_values if isinstance(value, str) and value and value in text))
