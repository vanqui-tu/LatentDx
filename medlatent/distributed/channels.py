"""Bounded structured message representation."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .messages import CommunicationCost, StructuredPayload, _payload_from_dict


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
