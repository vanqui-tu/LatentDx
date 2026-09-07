"""Bounded payload accounting for structured and text communication."""

from __future__ import annotations

import json

from .messages import Payload, TextProposalPayload, TextRequestPayload


class StructuredChannel:
    def __init__(self, *, max_wire_bytes: int = 4_096) -> None:
        if not isinstance(max_wire_bytes, int) or max_wire_bytes <= 0:
            raise ValueError("max_wire_bytes must be a positive integer")
        self.max_wire_bytes = max_wire_bytes

    def wire_bytes(self, payload: Payload) -> int:
        size = len(json.dumps(payload.to_dict(), separators=(",", ":"), sort_keys=True).encode("utf-8"))
        if size > self.max_wire_bytes:
            raise ValueError("structured payload exceeds max_wire_bytes")
        return size


class TextChannel(StructuredChannel):
    def __init__(self, *, max_tokens: int = 128, max_wire_bytes: int = 4_096) -> None:
        super().__init__(max_wire_bytes=max_wire_bytes)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens

    def wire_bytes(self, payload: Payload) -> int:
        if not isinstance(payload, (TextRequestPayload, TextProposalPayload)):
            raise TypeError("TextChannel accepts only text payloads")
        if len(payload.text.split()) > self.max_tokens:
            raise ValueError("text payload exceeds max_tokens")
        return super().wire_bytes(payload)
