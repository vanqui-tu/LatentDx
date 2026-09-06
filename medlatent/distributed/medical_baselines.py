"""Structured B0-B4 baselines over private medical stores."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .agent import AgentEpisodeState, AgentRuntime
from .aggregation import majority_vote, max_confidence, mean_score
from .episode import EpisodeEvent, EpisodeResult, SynchronousEpisodeEngine
from .graph import CommunicationGraph
from .medical import MedicalEpisode
from .messages import MessageEnvelope, MessageKind, ProposalPayload, RequestPayload
from .routing import DirectNeighborRouter, FloodUnvisitedRouter, PublicExpertiseRouter, RandomKRouter, Router


class MedicalBaselineKind(str, Enum):
    LOCAL_ONLY = "B0"
    ONE_HOP = "B1"
    FLOODING = "B2"
    RANDOM_K = "B3"
    HEURISTIC = "B4"


@dataclass(frozen=True, slots=True)
class MedicalTraceEvent:
    stage: str
    agent_id: int | None
    detail: str


@dataclass(frozen=True, slots=True)
class MedicalBaselineResult:
    kind: MedicalBaselineKind
    prediction: str | None
    episode: EpisodeResult
    contacted_agent_ids: tuple[int, ...]
    trace: tuple[MedicalTraceEvent, ...]

    @property
    def failure_stages(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                event.stage
                for event in self.trace
                if event.stage in {"unreachable", "routing", "aggregation"}
                or (event.stage == "retrieval" and "results=0" in event.detail)
            )
        )


def run_medical_baseline(
    episode: MedicalEpisode,
    graph: CommunicationGraph,
    agents: Mapping[int, AgentRuntime],
    kind: MedicalBaselineKind,
    *,
    max_rounds: int,
    max_fanout: int,
    seed: int = 0,
    public_expertise: Mapping[int, Sequence[str]] | None = None,
    aggregation: str = "mean_score",
    episode_id: str | None = None,
) -> MedicalBaselineResult:
    if episode.source_id not in agents or set(agents) != set(graph.agent_ids):
        raise ValueError("agents must exactly match graph IDs and include episode source")
    if episode_id is None:
        episode_id = f"medical:{episode.split}:{episode.query.case_id}:{kind.value}"
    aggregate = _aggregator(aggregation)
    trace: list[MedicalTraceEvent] = []
    local_proposals: list[ProposalPayload] = []
    router = _router(kind, episode, seed=seed, public_expertise=public_expertise)
    engine_rounds = max(1, max_rounds) if kind is MedicalBaselineKind.LOCAL_ONLY else max_rounds
    engine = SynchronousEpisodeEngine(graph, agents, max_rounds=engine_rounds, max_fanout=max_fanout)

    def handler(agent: AgentRuntime, state: AgentEpisodeState, round_index: int) -> Iterable[MessageEnvelope]:
        outgoing: list[MessageEnvelope] = []
        requests = tuple(message for message in state.current_inbox if message.kind is MessageKind.REQUEST)
        if agent.agent_id == episode.source_id and round_index == 0:
            proposal = _retrieve_proposal(agent, state, trace)
            if proposal is not None:
                local_proposals.append(proposal)
            if kind is not MedicalBaselineKind.LOCAL_ONLY:
                outgoing.extend(_requests(agent.agent_id, router, graph, episode_id, round_index, max_fanout, (agent.agent_id,)))
        if requests:
            proposal = _retrieve_proposal(agent, state, trace)
            if proposal is not None and state.parent_agent_id is not None:
                outgoing.append(_proposal(episode_id, round_index, agent.agent_id, state.parent_agent_id, proposal, requests[0].message_id))
            if kind in {MedicalBaselineKind.FLOODING, MedicalBaselineKind.RANDOM_K, MedicalBaselineKind.HEURISTIC}:
                request = requests[0]
                payload = request.payload
                assert isinstance(payload, RequestPayload)
                if request.ttl > 1:
                    request_budget = max(0, max_fanout - int(proposal is not None and state.parent_agent_id is not None))
                    outgoing.extend(_requests(agent.agent_id, router, graph, episode_id, round_index, request_budget, payload.visited_agent_ids + (agent.agent_id,)))
        if kind in {MedicalBaselineKind.FLOODING, MedicalBaselineKind.RANDOM_K, MedicalBaselineKind.HEURISTIC} and state.parent_agent_id is not None:
            for message in state.current_inbox:
                if message.kind is MessageKind.PROPOSAL and message.sender_id != state.parent_agent_id:
                    outgoing.append(_proposal(episode_id, round_index, agent.agent_id, state.parent_agent_id, message.payload, message.message_id))
        state.deactivate()
        return tuple(_round_trip(message) for message in outgoing)

    result = engine.run(
        episode_id=episode_id,
        source_id=episode.source_id,
        query=episode.query,
        handler=handler,
    )
    source_state = agents[episode.source_id].state_for(episode_id)
    proposals = tuple(local_proposals) + tuple(
        message.payload for message in source_state.received_messages if message.kind is MessageKind.PROPOSAL
    ) + tuple(message.payload for message in source_state.next_inbox if message.kind is MessageKind.PROPOSAL)
    aggregated = aggregate(proposals)
    prediction = None if aggregated is None else aggregated.candidate_label
    sent = tuple(event for event in result.events if event.event_type == "sent")
    contacted = tuple(sorted({event.receiver_id for event in sent if event.receiver_id is not None}))
    _classify_trace(trace, result.events, prediction, graph, episode.source_id, kind)
    if prediction is not None:
        trace.append(MedicalTraceEvent("success", episode.source_id, "proposal aggregated"))
    return MedicalBaselineResult(kind, prediction, result, contacted, tuple(trace))


def _router(
    kind: MedicalBaselineKind,
    episode: MedicalEpisode,
    *,
    seed: int,
    public_expertise: Mapping[int, Sequence[str]] | None,
) -> Router:
    if kind is MedicalBaselineKind.ONE_HOP:
        return DirectNeighborRouter()
    if kind is MedicalBaselineKind.RANDOM_K:
        return RandomKRouter(seed=seed)
    if kind is MedicalBaselineKind.HEURISTIC:
        return PublicExpertiseRouter(public_expertise or {}, episode.query.hpo_codes)
    return FloodUnvisitedRouter()


def _retrieve_proposal(agent: AgentRuntime, state: AgentEpisodeState, trace: list[MedicalTraceEvent]) -> ProposalPayload | None:
    records = agent.retrieve_active_episode(state.episode_id, limit=1)
    trace.append(MedicalTraceEvent("retrieval", agent.agent_id, f"results={len(records)}"))
    if not records:
        return None
    record = records[0]
    if float(record.score) <= 0.0:
        return None
    score = max(0.0, min(1.0, float(record.score)))
    return ProposalPayload(record.label, score, score)


def _requests(
    sender_id: int,
    router: Router,
    graph: CommunicationGraph,
    episode_id: str,
    round_index: int,
    max_fanout: int,
    visited: Sequence[int],
) -> tuple[MessageEnvelope, ...]:
    recipients = router.select_neighbors(graph, sender_id, max_fanout=max_fanout, excluded_agent_ids=visited, episode_id=episode_id, round_index=round_index)
    payload = RequestPayload("consult", visited_agent_ids=tuple(visited))
    return tuple(
        MessageEnvelope.create(
            message_id=f"{episode_id}:{round_index}:{sender_id}:{receiver_id}:REQUEST",
            episode_id=episode_id,
            round_sent=round_index,
            sender_id=sender_id,
            receiver_id=receiver_id,
            kind=MessageKind.REQUEST,
            ttl=max(1, graph.num_agents),
            payload=payload,
        )
        for receiver_id in recipients
    )


def _proposal(episode_id: str, round_index: int, sender_id: int, receiver_id: int, payload: ProposalPayload, parent_id: str) -> MessageEnvelope:
    return MessageEnvelope.create(
        message_id=f"{episode_id}:{round_index}:{sender_id}:{receiver_id}:PROPOSAL:{parent_id}",
        episode_id=episode_id,
        round_sent=round_index,
        sender_id=sender_id,
        receiver_id=receiver_id,
        kind=MessageKind.PROPOSAL,
        ttl=16,
        payload=payload,
        parent_message_id=parent_id,
    )


def _round_trip(message: MessageEnvelope) -> MessageEnvelope:
    return MessageEnvelope.create(
        message_id=message.message_id,
        episode_id=message.episode_id,
        round_sent=message.round_sent,
        sender_id=message.sender_id,
        receiver_id=message.receiver_id,
        kind=message.kind,
        ttl=message.ttl,
        payload=message.payload,
        parent_message_id=message.parent_message_id,
    )


def _aggregator(name: str):
    options = {"majority_vote": majority_vote, "mean_score": mean_score, "max_confidence": max_confidence}
    try:
        return options[name]
    except KeyError as error:
        raise ValueError(f"unknown medical aggregation: {name}") from error


def _classify_trace(
    trace: list[MedicalTraceEvent],
    events: Sequence[EpisodeEvent],
    prediction: str | None,
    graph: CommunicationGraph,
    source_id: int,
    kind: MedicalBaselineKind,
) -> None:
    sent = tuple(event for event in events if event.event_type == "sent")
    activated = tuple(event for event in events if event.event_type == "activated")
    request_sent = tuple(event for event in sent if event.message_id and ":REQUEST" in event.message_id)
    if len(graph.connected_components()) > 1:
        trace.append(MedicalTraceEvent("unreachable", source_id, "graph is disconnected"))
    if kind is not MedicalBaselineKind.LOCAL_ONLY and not request_sent:
        trace.append(MedicalTraceEvent("routing", source_id, "no request was sent"))
    if any(event.detail == "results=0" for event in trace if event.stage == "retrieval"):
        trace.append(MedicalTraceEvent("retrieval", None, "activated agent returned no candidate"))
    if activated and prediction is None:
        trace.append(MedicalTraceEvent("aggregation", source_id, "no proposal produced a prediction"))
