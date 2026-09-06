import pytest

from medlatent.distributed import (
    CommunicationCost,
    EvidencePayload,
    MessageEnvelope,
    MessageKind,
    ProposalPayload,
    RequestPayload,
    TextChannel,
    TextProposalPayload,
    TextRequestPayload,
    find_raw_substring_leaks,
)


def test_message_envelope_round_trips_with_structured_payload_and_cost():
    payload = ProposalPayload(candidate_label="DX:1", score=0.8, confidence=0.9)
    message = MessageEnvelope.create(
        message_id="m-1",
        episode_id="episode-1",
        round_sent=0,
        sender_id=0,
        receiver_id=1,
        kind=MessageKind.PROPOSAL,
        ttl=2,
        parent_message_id="request-1",
        payload=payload,
    )

    restored = MessageEnvelope.from_json(message.to_json())

    assert restored == message
    assert restored.cost == CommunicationCost.from_payload(payload)
    assert restored.parent_message_id == "request-1"


def test_payload_cost_is_deterministic_and_counts_utf8_bytes():
    payload = RequestPayload("consult", ("DX:1", "DX:2"))

    cost = CommunicationCost.from_payload(payload)

    assert cost.wire_bytes == cost.logical_size
    assert cost.wire_bytes == len(
        b'{"candidate_labels":["DX:1","DX:2"],"request_type":"consult","type":"request","visited_agent_ids":[]}'
    )


def test_message_rejects_mismatched_kind_and_payload():
    with pytest.raises(ValueError, match="does not match"):
        MessageEnvelope.create(
            message_id="m-1",
            episode_id="episode-1",
            round_sent=0,
            sender_id=0,
            receiver_id=1,
            kind=MessageKind.EVIDENCE,
            ttl=1,
            payload=ProposalPayload("DX:1", 1.0, 1.0),
        )


def test_evidence_payload_is_immutable():
    payload = EvidencePayload("DX:1", True, 1.0)

    with pytest.raises(AttributeError):
        payload.score = 0.0


def test_text_channel_round_trips_and_accounts_tokens_and_leaks():
    payload = TextProposalPayload("Likely diagnosis: private-disease.", "private-disease", 0.8, 0.8)
    channel = TextChannel(max_tokens=8, max_wire_bytes=256)

    encoded = channel.encode(payload)
    assert channel.decode(encoded) == payload
    assert channel.cost(payload).logical_size == 3
    assert find_raw_substring_leaks(payload.text, ("private-disease", "not-present")) == ("private-disease",)

    with pytest.raises(ValueError, match="max_tokens"):
        TextChannel(max_tokens=2).encode(payload)


def test_text_request_payload_is_typed_and_bounded():
    channel = TextChannel(max_tokens=6)
    payload = TextRequestPayload("Please return evidence.", (0, 2))
    assert channel.decode(channel.encode(payload)) == payload
