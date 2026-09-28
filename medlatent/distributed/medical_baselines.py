"""Tree-aggregated local, one-hop, and multi-hop medical baselines."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import random
from typing import Mapping, Sequence

from ..prompts import TEXTMAS_SYSTEM_PROMPT
from .agent import AgentRuntime
from .channels import TextChannel
from .episode import EpisodeEvent, EpisodeResult
from .graph import CommunicationGraph
from .medical import MedicalEpisode
from .messages import (
    MessageEnvelope,
    MessageKind,
    ProposalPayload,
    RequestPayload,
    TextProposalPayload,
    TextRequestPayload,
)
from .textmas import TextGenerator, build_agent_prompt, build_host_prompt, parse_textmas_answer


class MedicalBaselineKind(str, Enum):
    LOCAL_ONLY = "B0"
    ONE_HOP = "B1"
    FLOODING = "B2"


Proposal = ProposalPayload | TextProposalPayload


@dataclass(frozen=True, slots=True)
class MedicalBaselineResult:
    kind: MedicalBaselineKind
    prediction: str | None
    episode: EpisodeResult
    agents_reached: int
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
    max_text_wire_bytes: int = 4_096,
    text_generator: TextGenerator | None = None,
    max_text_completion_tokens: int = 128,
    seed: int = 42,
    episode_id: str | None = None,
) -> MedicalBaselineResult:
    """Run one bounded random query tree and aggregate replies bottom-up.

    ``max_rounds`` bounds query hops only. Replies are evaluated after the
    query tree is complete, so a depth-R leaf can still contribute to source.
    """
    if set(agents) != set(graph.agent_ids) or episode.source_id not in agents:
        raise ValueError("agents must exactly match graph IDs and include episode source")
    if channel not in {"structured", "text"}:
        raise ValueError("channel must be structured or text")
    if max_rounds < 0 or max_fanout < 0:
        raise ValueError("max_rounds and max_fanout must be non-negative")
    if max_text_tokens <= 0 or max_text_wire_bytes <= 0 or max_text_completion_tokens <= 0:
        raise ValueError("text limits must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    episode_id = episode_id or f"medical:{episode.split}:{episode.query.case_id}:{kind.value}:{channel}"
    depth_limit = 0 if kind is MedicalBaselineKind.LOCAL_ONLY else (1 if kind is MedicalBaselineKind.ONE_HOP else max_rounds)
    rng = random.Random(f"{seed}:{episode.split}:{episode.query.case_id}:{kind.value}:{episode.source_id}")
    text_channel = TextChannel(max_tokens=max_text_tokens, max_wire_bytes=max_text_wire_bytes) if channel == "text" else None
    events: list[EpisodeEvent] = []
    sent_payloads: list[object] = []
    usage = [0, 0]

    # Claim on request send: a query is one bounded random tree, not repeated
    # local retrieval from the same static private store through multiple paths.
    contexts: dict[int, dict[str, object]] = {
        episode.source_id: {"parent": None, "depth": 0, "children": []},
    }
    claimed = {episode.source_id}
    frontier = [episode.source_id]
    for depth in range(depth_limit):
        next_frontier: list[int] = []
        for agent_id in frontier:
            parent_id = contexts[agent_id]["parent"]
            candidates = [
                neighbor
                for neighbor in graph.neighbors(agent_id)
                if neighbor != parent_id and neighbor not in claimed
            ]
            for child_id in rng.sample(candidates, k=min(max_fanout, len(candidates))):
                claimed.add(child_id)
                contexts[child_id] = {"parent": agent_id, "depth": depth + 1, "children": []}
                contexts[agent_id]["children"].append(child_id)  # type: ignore[index]
                _record_message(
                    events,
                    sent_payloads,
                    _request(episode_id, depth, agent_id, child_id, channel),
                    text_channel,
                )
                next_frontier.append(child_id)
        frontier = next_frontier

    replies: dict[int, Proposal | None] = {}
    for agent_id in sorted(contexts, key=lambda node: (-int(contexts[node]["depth"]), node)):
        local = _retrieve(
            agents[agent_id], episode_id, episode.query, channel, text_generator,
            max_text_completion_tokens, usage,
        )
        child_replies = [replies[child_id] for child_id in contexts[agent_id]["children"]]  # type: ignore[index]
        payload = _aggregate(
            local,
            child_replies,
            channel=channel,
            query_text=episode.query.phenotype_text,
            text_generator=text_generator,
            max_text_completion_tokens=max_text_completion_tokens,
            usage=usage,
        )
        replies[agent_id] = payload
        parent_id = contexts[agent_id]["parent"]
        if parent_id is not None and payload is not None:
            _record_message(
                events,
                sent_payloads,
                _proposal(
                    episode_id,
                    int(contexts[agent_id]["depth"]),
                    agent_id,
                    int(parent_id),
                    payload,
                ),
                text_channel,
            )

    root_payload = replies[episode.source_id]
    return MedicalBaselineResult(
        kind=kind,
        prediction=root_payload.candidate_label if root_payload is not None else None,
        episode=EpisodeResult(episode_id, episode.source_id, "completed", tuple(events)),
        agents_reached=len(contexts),
        text_tokens=sum(
            _text_tokens(payload.text)
            for payload in sent_payloads
            if isinstance(payload, (TextRequestPayload, TextProposalPayload))
        ),
        wire_bytes=sum(event.wire_bytes for event in events if event.event_type == "sent"),
        text_prompt_tokens=usage[0],
        text_completion_tokens=usage[1],
    )


def _request(
    episode_id: str, depth: int, sender_id: int, receiver_id: int, channel: str,
) -> MessageEnvelope:
    payload = TextRequestPayload("Return an aggregated diagnostic summary.") if channel == "text" else RequestPayload()
    return MessageEnvelope.create(
        message_id=f"{episode_id}:request:{depth}:{sender_id}:{receiver_id}",
        episode_id=episode_id,
        round_sent=depth,
        sender_id=sender_id,
        receiver_id=receiver_id,
        kind=MessageKind.REQUEST,
        payload=payload,
    )


def _proposal(
    episode_id: str, depth: int, sender_id: int, receiver_id: int, payload: Proposal,
) -> MessageEnvelope:
    return MessageEnvelope.create(
        message_id=f"{episode_id}:proposal:{depth}:{sender_id}:{receiver_id}",
        episode_id=episode_id,
        round_sent=depth,
        sender_id=sender_id,
        receiver_id=receiver_id,
        kind=MessageKind.PROPOSAL,
        payload=payload,
    )


def _record_message(
    events: list[EpisodeEvent],
    payloads: list[object],
    message: MessageEnvelope,
    text_channel: TextChannel | None,
) -> None:
    if text_channel is not None:
        text_channel.wire_bytes(message.payload)
    payloads.append(message.payload)
    events.append(EpisodeEvent(
        round_index=message.round_sent,
        event_type="sent",
        agent_id=message.sender_id,
        message_id=message.message_id,
        receiver_id=message.receiver_id,
        wire_bytes=message.wire_bytes,
    ))


def _retrieve(
    agent: AgentRuntime,
    episode_id: str,
    query: object,
    channel: str,
    text_generator: TextGenerator | None,
    max_new_tokens: int,
    usage: list[int],
) -> Proposal | None:
    state = agent.reset_episode(episode_id, query)
    state.processing = True
    try:
        records = agent.retrieve_active_episode(episode_id, limit=1)
    finally:
        state.processing = False
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
            hospital_id=agent.agent_id,
            case_disease=record.label,
            case_phenotype=record.phenotype_text,
            test_phenotype=state.query.phenotype_text,
        ),
        max_new_tokens=max_new_tokens,
    )
    usage[0] += generated.prompt_tokens
    usage[1] += generated.completion_tokens
    return TextProposalPayload(generated.text, parse_textmas_answer(generated.text) or record.label, score)


def _aggregate(
    local: Proposal | None,
    child_replies: Sequence[Proposal | None],
    *,
    channel: str,
    query_text: str,
    text_generator: TextGenerator | None,
    max_text_completion_tokens: int,
    usage: list[int],
) -> Proposal | None:
    child_payloads = [payload for payload in child_replies if payload is not None]
    proposals = ([local] if local is not None else []) + child_payloads
    structured = _aggregate_structured(proposals)
    if structured is None:
        return None
    if channel != "text":
        return structured
    if not child_payloads and isinstance(local, TextProposalPayload):
        return local

    texts = [payload.text for payload in proposals if isinstance(payload, TextProposalPayload)]
    if text_generator is None:
        return TextProposalPayload(
            f"Likely diagnosis: {structured.candidate_label}.",
            structured.candidate_label,
            structured.score,
        )
    generated = text_generator.generate(
        TEXTMAS_SYSTEM_PROMPT,
        build_host_prompt(context="\n".join(texts), test_phenotype=query_text),
        max_new_tokens=max_text_completion_tokens,
    )
    usage[0] += generated.prompt_tokens
    usage[1] += generated.completion_tokens
    return TextProposalPayload(
        generated.text,
        parse_textmas_answer(generated.text) or structured.candidate_label,
        structured.score,
    )


def _aggregate_structured(proposals: Sequence[Proposal]) -> ProposalPayload | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for proposal in proposals:
        grouped[proposal.candidate_label].append(proposal.score)
    if not grouped:
        return None
    label = max(grouped, key=lambda value: (sum(grouped[value]) / len(grouped[value]), value))
    return ProposalPayload(label, sum(grouped[label]) / len(grouped[label]))


def _text_tokens(text: str) -> int:
    return len(text.split())
