"""Structured B0-B2 baselines on the common synchronous episode engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .agent import AgentEpisodeState, AgentRuntime
from .aggregation import AggregatedProposal, require_all_evidence
from .channels import StructuredChannel
from .episode import EpisodeResult, SynchronousEpisodeEngine
from .fixtures import ComplementaryEvidenceFixture
from .messages import MessageEnvelope, MessageKind, ProposalPayload, RequestPayload
from .routing import DirectNeighborRouter, FloodUnvisitedRouter


class BaselineKind(str, Enum):
    LOCAL_ONLY = "B0"
    ONE_HOP = "B1"
    FLOODING = "B2"


@dataclass(frozen=True, slots=True)
class BaselineResult:
    kind: BaselineKind
    prediction: str | None
    episode: EpisodeResult | None
    contacted_agent_ids: tuple[int, ...]


def run_baseline(
    fixture: ComplementaryEvidenceFixture,
    kind: BaselineKind,
    *,
    max_rounds: int,
    max_fanout: int,
    episode_id: str = "synthetic-episode",
) -> BaselineResult:
    agents = fixture.build_agents()
    source = agents[fixture.source_id]
    if kind is BaselineKind.LOCAL_ONLY:
        prediction = _predict(fixture, _local_proposals(source, fixture.query))
        return BaselineResult(kind, prediction, None, ())

    engine = SynchronousEpisodeEngine(
        fixture.graph,
        agents,
        max_rounds=max_rounds,
        max_fanout=max_fanout,
    )
    channel = StructuredChannel()
    router = DirectNeighborRouter() if kind is BaselineKind.ONE_HOP else FloodUnvisitedRouter()
    handler = _handler(fixture, kind, router, channel, max_fanout, episode_id)
    episode = engine.run(
        episode_id=episode_id,
        source_id=fixture.source_id,
        query=fixture.query,
        handler=handler,
    )
    source_state = source.state_for(episode_id)
    proposals = _proposals(source_state.received_messages) + _proposals(source_state.next_inbox)
    prediction = _predict(fixture, proposals)
    contacted = tuple(
        sorted({event.receiver_id for event in episode.events if event.event_type == "sent" and event.receiver_id is not None})
    )
    return BaselineResult(kind, prediction, episode, contacted)


def _handler(
    fixture: ComplementaryEvidenceFixture,
    kind: BaselineKind,
    router: DirectNeighborRouter,
    channel: StructuredChannel,
    max_fanout: int,
    episode_id: str,
):
    def handle(agent: AgentRuntime, state: AgentEpisodeState, round_index: int) -> Iterable[MessageEnvelope]:
        requests = tuple(message for message in state.current_inbox if message.kind is MessageKind.REQUEST)
        outgoing: list[MessageEnvelope] = []
        if agent.agent_id == fixture.source_id and round_index == 0:
            outgoing.extend(
                _requests_to_neighbors(
                    fixture,
                    router,
                    agent.agent_id,
                    state,
                    round_index,
                    max_fanout,
                    episode_id,
                    visited_agent_ids=(agent.agent_id,),
                )
            )
        if requests:
            proposal = _local_proposal(agent, fixture.query)
            if proposal is not None and state.parent_agent_id is not None:
                outgoing.append(
                    _message(
                        episode_id,
                        round_index,
                        agent.agent_id,
                        state.parent_agent_id,
                        MessageKind.PROPOSAL,
                        proposal,
                    )
                )
            if kind is BaselineKind.FLOODING:
                request = requests[0]
                payload = request.payload
                assert isinstance(payload, RequestPayload)
                outgoing.extend(
                    _requests_to_neighbors(
                        fixture,
                        router,
                        agent.agent_id,
                        state,
                        round_index,
                        max_fanout,
                        episode_id,
                        visited_agent_ids=payload.visited_agent_ids + (agent.agent_id,),
                    )
                )
        if kind is BaselineKind.FLOODING and state.parent_agent_id is not None:
            for message in state.current_inbox:
                if message.kind is MessageKind.PROPOSAL and message.sender_id != state.parent_agent_id:
                    outgoing.append(
                        _message(
                            episode_id,
                            round_index,
                            agent.agent_id,
                            state.parent_agent_id,
                            MessageKind.PROPOSAL,
                            message.payload,
                        )
                    )
        channel_messages = []
        for message in outgoing:
            encoded = channel.encode(message.payload)
            payload = channel.decode(encoded)
            channel_messages.append(
                MessageEnvelope.create(
                    message_id=message.message_id,
                    episode_id=message.episode_id,
                    round_sent=message.round_sent,
                    sender_id=message.sender_id,
                    receiver_id=message.receiver_id,
                    kind=message.kind,
                    ttl=message.ttl,
                    payload=payload,
                    parent_message_id=message.parent_message_id,
                )
            )
        state.deactivate()
        return tuple(channel_messages)

    return handle


def _requests_to_neighbors(
    fixture: ComplementaryEvidenceFixture,
    router: DirectNeighborRouter,
    sender_id: int,
    state: AgentEpisodeState,
    round_index: int,
    max_fanout: int,
    episode_id: str,
    *,
    visited_agent_ids: tuple[int, ...],
) -> tuple[MessageEnvelope, ...]:
    recipients = router.select_neighbors(
        fixture.graph,
        sender_id,
        max_fanout=max_fanout,
        excluded_agent_ids=visited_agent_ids,
        episode_id=episode_id,
        round_index=round_index,
    )
    payload = RequestPayload("consult", visited_agent_ids=visited_agent_ids)
    return tuple(
        _message(episode_id, round_index, sender_id, receiver_id, MessageKind.REQUEST, payload)
        for receiver_id in recipients
    )


def _message(
    episode_id: str,
    round_index: int,
    sender_id: int,
    receiver_id: int,
    kind: MessageKind,
    payload: RequestPayload | ProposalPayload,
) -> MessageEnvelope:
    message_id = f"{episode_id}:{round_index}:{sender_id}:{receiver_id}:{kind.value}:{payload.to_dict()}"
    return MessageEnvelope.create(
        message_id=message_id,
        episode_id=episode_id,
        round_sent=round_index,
        sender_id=sender_id,
        receiver_id=receiver_id,
        kind=kind,
        ttl=16,
        payload=payload,
    )


def _local_proposals(agent: AgentRuntime, query: object) -> tuple[ProposalPayload, ...]:
    return tuple(
        ProposalPayload(record.label, record.score, 1.0)
        for record in agent.retrieve_local(query)
    )


def _local_proposal(agent: AgentRuntime, query: object) -> ProposalPayload | None:
    proposals = _local_proposals(agent, query)
    return proposals[0] if proposals else None


def _proposals(messages: Iterable[MessageEnvelope]) -> tuple[ProposalPayload, ...]:
    return tuple(message.payload for message in messages if message.kind is MessageKind.PROPOSAL)


def _predict(fixture: ComplementaryEvidenceFixture, proposals: Iterable[ProposalPayload]) -> str | None:
    aggregated: AggregatedProposal | None = require_all_evidence(
        proposals,
        target_label=fixture.target_label,
        required_evidence_labels=fixture.required_evidence_labels,
    )
    return None if aggregated is None else aggregated.candidate_label
