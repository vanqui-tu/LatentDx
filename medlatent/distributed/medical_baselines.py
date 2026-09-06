"""Structured B0-B4 baselines over private medical stores."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence

from .agent import AgentEpisodeState, AgentRuntime
from .aggregation import majority_vote, max_confidence, mean_score
from .channels import TextChannel, find_raw_substring_leaks
from .episode import EpisodeEvent, EpisodeResult, SynchronousEpisodeEngine
from .graph import CommunicationGraph
from .medical import MedicalEpisode
from .messages import MessageEnvelope, MessageKind, ProposalPayload, RequestPayload, TextProposalPayload, TextRequestPayload
from .routing import DirectNeighborRouter, FloodUnvisitedRouter, PublicExpertiseRouter, RandomKRouter, Router
from .textmas import TextGenerator, build_agent_prompt, build_host_prompt, parse_textmas_answer
from ..prompts import TEXTMAS_SYSTEM_PROMPT


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
    text_tokens: int = 0
    wire_bytes: int = 0
    leaked_substrings: tuple[str, ...] = ()
    text_prompt_tokens: int = 0
    text_completion_tokens: int = 0

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
    channel: str = "structured",
    max_text_tokens: int = 128,
    max_text_wire_bytes: int = 4_096,
    private_values: Sequence[str] = (),
    text_aggregator: Callable[[tuple[str, ...]], str | None] | None = None,
    text_generator: TextGenerator | None = None,
    max_text_completion_tokens: int = 128,
    episode_id: str | None = None,
) -> MedicalBaselineResult:
    if episode.source_id not in agents or set(agents) != set(graph.agent_ids):
        raise ValueError("agents must exactly match graph IDs and include episode source")
    if episode_id is None:
        episode_id = f"medical:{episode.split}:{episode.query.case_id}:{kind.value}"
    aggregate = _aggregator(aggregation)
    if channel not in {"structured", "text"}:
        raise ValueError("unknown medical channel")
    text_channel = TextChannel(max_tokens=max_text_tokens, max_wire_bytes=max_text_wire_bytes) if channel == "text" else None
    trace: list[MedicalTraceEvent] = []
    local_proposals: list[ProposalPayload] = []
    local_texts: list[str] = []
    text_tokens = 0
    text_prompt_tokens = 0
    text_completion_tokens = 0
    generation_usage = [0, 0]
    leaked: list[str] = []
    router = _router(kind, episode, seed=seed, public_expertise=public_expertise)
    engine_rounds = max(1, max_rounds) if kind is MedicalBaselineKind.LOCAL_ONLY else max_rounds
    engine = SynchronousEpisodeEngine(graph, agents, max_rounds=engine_rounds, max_fanout=max_fanout)

    def handler(agent: AgentRuntime, state: AgentEpisodeState, round_index: int) -> Iterable[MessageEnvelope]:
        nonlocal text_tokens
        outgoing: list[MessageEnvelope] = []
        requests = tuple(message for message in state.current_inbox if message.kind is MessageKind.REQUEST)
        if agent.agent_id == episode.source_id and round_index == 0:
            proposal = _retrieve_proposal(
                agent, state, trace, text=channel == "text", text_generator=text_generator,
                max_text_completion_tokens=max_text_completion_tokens, generation_usage=generation_usage,
            )
            if proposal is not None:
                local_proposals.append(_proposal_value(proposal))
                if isinstance(proposal, TextProposalPayload):
                    local_texts.append(proposal.text)
            if kind is not MedicalBaselineKind.LOCAL_ONLY:
                outgoing.extend(_requests(agent.agent_id, router, graph, episode_id, round_index, max_fanout, (agent.agent_id,), text=channel == "text"))
        if requests:
            proposal = _retrieve_proposal(
                agent, state, trace, text=channel == "text", text_generator=text_generator,
                max_text_completion_tokens=max_text_completion_tokens, generation_usage=generation_usage,
            )
            if proposal is not None and state.parent_agent_id is not None:
                outgoing.append(_proposal(episode_id, round_index, agent.agent_id, state.parent_agent_id, proposal, requests[0].message_id))
            if kind in {MedicalBaselineKind.FLOODING, MedicalBaselineKind.RANDOM_K, MedicalBaselineKind.HEURISTIC}:
                request = requests[0]
                payload = request.payload
                assert isinstance(payload, (RequestPayload, TextRequestPayload))
                if request.ttl > 1:
                    request_budget = max(0, max_fanout - int(proposal is not None and state.parent_agent_id is not None))
                    outgoing.extend(_requests(agent.agent_id, router, graph, episode_id, round_index, request_budget, payload.visited_agent_ids + (agent.agent_id,), text=channel == "text"))
        if kind in {MedicalBaselineKind.FLOODING, MedicalBaselineKind.RANDOM_K, MedicalBaselineKind.HEURISTIC} and state.parent_agent_id is not None:
            for message in state.current_inbox:
                if message.kind is MessageKind.PROPOSAL and message.sender_id != state.parent_agent_id:
                    outgoing.append(_proposal(episode_id, round_index, agent.agent_id, state.parent_agent_id, message.payload, message.message_id))
        state.deactivate()
        for message in outgoing:
            if text_channel is not None and isinstance(message.payload, (TextRequestPayload, TextProposalPayload)):
                encoded = text_channel.encode(message.payload)
                decoded = text_channel.decode(encoded)
                text_tokens += text_channel.cost(decoded).logical_size
                leaked.extend(find_raw_substring_leaks(decoded.text, private_values))
        return tuple(_round_trip(message, text_channel) for message in outgoing)

    result = engine.run(
        episode_id=episode_id,
        source_id=episode.source_id,
        query=episode.query,
        handler=handler,
    )
    source_state = agents[episode.source_id].state_for(episode_id)
    proposals = tuple(local_proposals) + tuple(
        _proposal_value(message.payload) for message in source_state.received_messages if message.kind is MessageKind.PROPOSAL
    ) + tuple(_proposal_value(message.payload) for message in source_state.next_inbox if message.kind is MessageKind.PROPOSAL)
    text_proposals = tuple(local_texts) + tuple(
        message.payload.text
        for message in source_state.received_messages
        if message.kind is MessageKind.PROPOSAL and isinstance(message.payload, TextProposalPayload)
    ) + tuple(
        message.payload.text
        for message in source_state.next_inbox
        if message.kind is MessageKind.PROPOSAL and isinstance(message.payload, TextProposalPayload)
    )
    if channel == "text" and text_aggregator is not None:
        prediction = text_aggregator(text_proposals)
    elif channel == "text" and text_generator is not None:
        generated = text_generator.generate(
            TEXTMAS_SYSTEM_PROMPT,
            build_host_prompt(context="\n".join(text_proposals), test_phenotype=episode.query.phenotype_text),
            max_new_tokens=max_text_completion_tokens,
        )
        generation_usage[0] += generated.prompt_tokens
        generation_usage[1] += generated.completion_tokens
        prediction = parse_textmas_answer(generated.text)
    else:
        aggregated = aggregate(proposals)
        prediction = None if aggregated is None else aggregated.candidate_label
    sent = tuple(event for event in result.events if event.event_type == "sent")
    contacted = tuple(sorted({event.receiver_id for event in sent if event.receiver_id is not None}))
    _classify_trace(trace, result.events, prediction, graph, episode.source_id, kind)
    if prediction is not None:
        trace.append(MedicalTraceEvent("success", episode.source_id, "proposal aggregated"))
    text_prompt_tokens = generation_usage[0]
    text_completion_tokens = generation_usage[1]
    return MedicalBaselineResult(
        kind, prediction, result, contacted, tuple(trace), text_tokens,
        sum(event.wire_bytes for event in sent), tuple(dict.fromkeys(leaked)),
        text_prompt_tokens, text_completion_tokens,
    )


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


def _retrieve_proposal(
    agent: AgentRuntime,
    state: AgentEpisodeState,
    trace: list[MedicalTraceEvent],
    *,
    text: bool = False,
    text_generator: TextGenerator | None = None,
    max_text_completion_tokens: int = 128,
    generation_usage: list[int] | None = None,
) -> ProposalPayload | TextProposalPayload | None:
    records = agent.retrieve_active_episode(state.episode_id, limit=1)
    trace.append(MedicalTraceEvent("retrieval", agent.agent_id, f"results={len(records)}"))
    if not records:
        return None
    record = records[0]
    if float(record.score) <= 0.0:
        return None
    score = max(0.0, min(1.0, float(record.score)))
    if text:
        if text_generator is not None:
            generated = text_generator.generate(
                TEXTMAS_SYSTEM_PROMPT,
                build_agent_prompt(
                    hospital_id=agent.agent_id,
                    case_disease=record.label,
                    case_phenotype=record.phenotype_text,
                    test_phenotype=state.query.phenotype_text,
                ),
                max_new_tokens=max_text_completion_tokens,
            )
            if generation_usage is not None:
                generation_usage[0] += generated.prompt_tokens
                generation_usage[1] += generated.completion_tokens
            candidate = parse_textmas_answer(generated.text) or record.label
            return TextProposalPayload(generated.text, candidate, score, score)
        return TextProposalPayload(
            f"Likely diagnosis: {record.label}. Retrieval score: {score:.4f}.",
            record.label,
            score,
            score,
        )
    return ProposalPayload(record.label, score, score)


def _requests(
    sender_id: int,
    router: Router,
    graph: CommunicationGraph,
    episode_id: str,
    round_index: int,
    max_fanout: int,
    visited: Sequence[int],
    *,
    text: bool = False,
) -> tuple[MessageEnvelope, ...]:
    recipients = router.select_neighbors(graph, sender_id, max_fanout=max_fanout, excluded_agent_ids=visited, episode_id=episode_id, round_index=round_index)
    payload = (
        TextRequestPayload("Please return a bounded diagnostic summary.", tuple(visited))
        if text
        else RequestPayload("consult", visited_agent_ids=tuple(visited))
    )
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


def _proposal(
    episode_id: str,
    round_index: int,
    sender_id: int,
    receiver_id: int,
    payload: ProposalPayload | TextProposalPayload,
    parent_id: str,
) -> MessageEnvelope:
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


def _round_trip(message: MessageEnvelope, text_channel: TextChannel | None = None) -> MessageEnvelope:
    cost = text_channel.cost(message.payload) if text_channel is not None else None
    return MessageEnvelope.create(
        message_id=message.message_id,
        episode_id=message.episode_id,
        round_sent=message.round_sent,
        sender_id=message.sender_id,
        receiver_id=message.receiver_id,
        kind=message.kind,
        ttl=message.ttl,
        payload=message.payload,
        cost=cost,
        parent_message_id=message.parent_message_id,
    )


def _proposal_value(payload: ProposalPayload | TextProposalPayload) -> ProposalPayload:
    if isinstance(payload, ProposalPayload):
        return payload
    return ProposalPayload(payload.candidate_label, payload.score, payload.confidence)


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
