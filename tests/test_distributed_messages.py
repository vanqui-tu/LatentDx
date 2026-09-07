import pytest

from medlatent.distributed import MessageEnvelope, MessageKind, ProposalPayload, RequestPayload, StructuredChannel, TextChannel, TextProposalPayload


def test_envelope_charges_payload_bytes_without_serializing_transport():
    message = MessageEnvelope.create(
        message_id="m1", episode_id="e1", round_sent=0, sender_id=0, receiver_id=1,
        kind=MessageKind.PROPOSAL, payload=ProposalPayload("DX:1", 0.8),
    )
    assert message.wire_bytes == StructuredChannel().wire_bytes(message.payload)
    with pytest.raises(ValueError, match="does not match"):
        MessageEnvelope.create(
            message_id="bad", episode_id="e1", round_sent=0, sender_id=0, receiver_id=1,
            kind=MessageKind.REQUEST, payload=ProposalPayload("DX:1", 0.8),
        )


def test_text_channel_enforces_token_limit():
    payload = TextProposalPayload("one two three", "DX:1", 0.8)
    assert TextChannel(max_tokens=3).wire_bytes(payload) > 0
    with pytest.raises(ValueError, match="max_tokens"):
        TextChannel(max_tokens=2).wire_bytes(payload)
    assert RequestPayload((0, 2)).visited_agent_ids == (0, 2)
