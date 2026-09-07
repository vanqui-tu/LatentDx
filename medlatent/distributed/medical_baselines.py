"""M2 medical baselines: local, one-hop, and multi-hop flooding."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence

from ..prompts import TEXTMAS_SYSTEM_PROMPT
from .agent import AgentEpisodeState, AgentRuntime
from .episode import EpisodeResult, SynchronousEpisodeEngine
from .graph import CommunicationGraph
from .medical import MedicalEpisode
from .messages import (
    MessageEnvelope, MessageKind, Proposal, ProposalPayload, Request, RequestPayload,
    TextProposalPayload, TextRequestPayload,
)
from .textmas import TextGenerator, build_agent_prompt, build_host_prompt, parse_textmas_answer


class MedicalBaselineKind(str, Enum):
    LOCAL_ONLY = "B0"
    ONE_HOP = "B1"
    FLOODING = "B2"


@dataclass(frozen=True, slots=True)
class MedicalBaselineResult:
    kind: MedicalBaselineKind
    prediction: str | None
    episode: EpisodeResult
    contacted_agent_ids: tuple[int, ...]
    text_tokens: int
    wire_bytes: int
    text_prompt_tokens: int = 0
    text_completion_tokens: int = 0


def run_medical_baseline(
    episode: MedicalEpisode,
    graph: CommunicationGraph,
    agents: Mapping[int, AgentRuntime],
    kind: MedicalBaselineKind,
    *,
    max_rounds: int,
    max_fanout: int,
    channel: str = "structured",
    max_text_tokens: int = 128,
    text_aggregator: Callable[[tuple[str, ...]], str | None] | None = None,
    text_generator: TextGenerator | None = None,
    max_text_completion_tokens: int = 128,
    episode_id: str | None = None,
) -> MedicalBaselineResult:
    if set(agents) != set(graph.agent_ids) or episode.source_id not in agents:
        raise ValueError("agents must exactly match graph IDs and include episode source")
    if channel not in {"structured", "text"}:
        raise ValueError("channel must be structured or text")
    if max_text_tokens <= 0:
        raise ValueError("max_text_tokens must be positive")
    episode_id = episode_id or f"medical:{episode.split}:{episode.query.case_id}:{kind.value}:{channel}"
    local_proposals: list[ProposalPayload] = []
    local_texts: list[str] = []
    usage = [0, 0]
    text_tokens = 0

    def handler(agent: AgentRuntime, state: AgentEpisodeState, round_index: int) -> Iterable[MessageEnvelope]:
        nonlocal text_tokens
        outgoing: list[MessageEnvelope] = []
        requests = tuple(message for message in state.current_inbox if message.kind is MessageKind.REQUEST)
        if agent.agent_id == episode.source_id and round_index == 0:
            proposal = _retrieve(agent, state, channel, text_generator, max_text_completion_tokens, usage)
            if proposal is not None:
                local_proposals.append(_structured(proposal))
                if isinstance(proposal, TextProposalPayload):
                    local_texts.append(proposal.text)
            if kind is not MedicalBaselineKind.LOCAL_ONLY:
                outgoing.extend(_requests(agent.agent_id, graph, episode_id, round_index, max_fanout, (agent.agent_id,), channel))

        if requests:
            request = requests[0]
            proposal = _retrieve(agent, state, channel, text_generator, max_text_completion_tokens, usage)
            if proposal is not None and state.parent_agent_id is not None:
                outgoing.append(_proposal(episode_id, round_index, agent.agent_id, state.parent_agent_id, proposal))
            if kind is MedicalBaselineKind.FLOODING:
                visited = _visited(request.payload) + (agent.agent_id,)
                fanout = max(0, max_fanout - int(proposal is not None and state.parent_agent_id is not None))
                outgoing.extend(_requests(agent.agent_id, graph, episode_id, round_index, fanout, visited, channel))

        if kind is MedicalBaselineKind.FLOODING and state.parent_agent_id is not None:
            for message in state.current_inbox:
                if message.kind is MessageKind.PROPOSAL and message.sender_id != state.parent_agent_id:
                    outgoing.append(_proposal(episode_id, round_index, agent.agent_id, state.parent_agent_id, message.payload))
        state.deactivate()
        _validate_text(outgoing, max_text_tokens)
        text_tokens += sum(
            _text_tokens(message.payload.text)
            for message in outgoing
            if isinstance(message.payload, (TextRequestPayload, TextProposalPayload))
        )
        return outgoing

    engine = SynchronousEpisodeEngine(
        graph,
        agents,
        max_rounds=max(1, max_rounds) if kind is MedicalBaselineKind.LOCAL_ONLY else max_rounds,
        max_fanout=max_fanout,
    )
    result = engine.run(episode_id=episode_id, source_id=episode.source_id, query=episode.query, handler=handler)
    source = agents[episode.source_id].state_for(episode_id)
    received = tuple(source.received_messages) + tuple(source.next_inbox)
    proposals = tuple(local_proposals) + tuple(
        _structured(message.payload) for message in received if message.kind is MessageKind.PROPOSAL
    )
    texts = tuple(local_texts) + tuple(
        message.payload.text
        for message in received
        if message.kind is MessageKind.PROPOSAL and isinstance(message.payload, TextProposalPayload)
    )
    if channel == "text" and text_aggregator is not None:
        prediction = text_aggregator(texts)
    elif channel == "text" and text_generator is not None:
        generated = text_generator.generate(
            TEXTMAS_SYSTEM_PROMPT,
            build_host_prompt(context="\n".join(texts), test_phenotype=episode.query.phenotype_text),
            max_new_tokens=max_text_completion_tokens,
        )
        usage[0] += generated.prompt_tokens
        usage[1] += generated.completion_tokens
        prediction = parse_textmas_answer(generated.text)
    else:
        prediction = _mean_score(proposals)
    sent = tuple(event for event in result.events if event.event_type == "sent")
    return MedicalBaselineResult(
        kind, prediction, result,
        tuple(sorted({event.receiver_id for event in sent if event.receiver_id is not None})),
        text_tokens, sum(event.wire_bytes for event in sent), usage[0], usage[1],
    )


def _requests(
    sender_id: int, graph: CommunicationGraph, episode_id: str, round_index: int,
    max_fanout: int, visited: Sequence[int], channel: str,
) -> tuple[MessageEnvelope, ...]:
    excluded = set(visited)
    recipients = tuple(neighbor for neighbor in graph.neighbors(sender_id) if neighbor not in excluded)[:max_fanout]
    payload: Request = (
        TextRequestPayload("Return a bounded diagnostic summary.", tuple(visited))
        if channel == "text" else RequestPayload(tuple(visited))
    )
    return tuple(
        MessageEnvelope.create(
            message_id=f"{episode_id}:{round_index}:{sender_id}:{receiver_id}:request",
            episode_id=episode_id, round_sent=round_index, sender_id=sender_id, receiver_id=receiver_id,
            kind=MessageKind.REQUEST, payload=payload,
        )
        for receiver_id in recipients
    )


def _proposal(
    episode_id: str, round_index: int, sender_id: int, receiver_id: int, payload: Proposal,
) -> MessageEnvelope:
    return MessageEnvelope.create(
        message_id=f"{episode_id}:{round_index}:{sender_id}:{receiver_id}:proposal",
        episode_id=episode_id, round_sent=round_index, sender_id=sender_id, receiver_id=receiver_id,
        kind=MessageKind.PROPOSAL, payload=payload,
    )


def _retrieve(
    agent: AgentRuntime, state: AgentEpisodeState, channel: str, text_generator: TextGenerator | None,
    max_new_tokens: int, usage: list[int],
) -> Proposal | None:
    records = agent.retrieve_active_episode(state.episode_id, limit=1)
    if not records or float(records[0].score) <= 0.0:
        return None
    record = records[0]
    score = min(1.0, float(record.score))
    if channel != "text":
        return ProposalPayload(record.label, score)
    if text_generator is None:
        return TextProposalPayload(f"Likely diagnosis: {record.label}.", record.label, score)
    generated = text_generator.generate(
        TEXTMAS_SYSTEM_PROMPT,
        build_agent_prompt(
            hospital_id=agent.agent_id, case_disease=record.label,
            case_phenotype=record.phenotype_text, test_phenotype=state.query.phenotype_text,
        ),
        max_new_tokens=max_new_tokens,
    )
    usage[0] += generated.prompt_tokens
    usage[1] += generated.completion_tokens
    return TextProposalPayload(generated.text, parse_textmas_answer(generated.text) or record.label, score)


def _mean_score(proposals: Sequence[ProposalPayload]) -> str | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for proposal in proposals:
        grouped[proposal.candidate_label].append(proposal.score)
    if not grouped:
        return None
    return max(grouped, key=lambda label: (sum(grouped[label]) / len(grouped[label]), label))


def _structured(payload: Proposal) -> ProposalPayload:
    return payload if isinstance(payload, ProposalPayload) else ProposalPayload(payload.candidate_label, payload.score)


def _visited(payload: object) -> tuple[int, ...]:
    assert isinstance(payload, (RequestPayload, TextRequestPayload))
    return payload.visited_agent_ids


def _validate_text(messages: Sequence[MessageEnvelope], max_tokens: int) -> None:
    if any(
        _text_tokens(message.payload.text) > max_tokens
        for message in messages
        if isinstance(message.payload, (TextRequestPayload, TextProposalPayload))
    ):
        raise ValueError("text payload exceeds max_text_tokens")


def _text_tokens(text: str) -> int:
    return len(text.split())
