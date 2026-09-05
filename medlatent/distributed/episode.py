"""Local synchronous episode scheduler for distributed communication."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from .agent import AgentEpisodeState, AgentRuntime
from .graph import CommunicationGraph
from .messages import MessageEnvelope


RoundHandler = Callable[[AgentRuntime, AgentEpisodeState, int], Iterable[MessageEnvelope]]
StopRule = Callable[[AgentRuntime, AgentEpisodeState, int], bool]


@dataclass(frozen=True, slots=True)
class EpisodeEvent:
    round_index: int
    event_type: str
    agent_id: int
    message_id: str | None = None
    receiver_id: int | None = None
    wire_bytes: int = 0


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    episode_id: str
    source_id: int
    termination_reason: str
    events: tuple[EpisodeEvent, ...]


class SynchronousEpisodeEngine:
    """Run query-scoped local agents with next-round message delivery."""

    def __init__(
        self,
        graph: CommunicationGraph,
        agents: Mapping[int, AgentRuntime],
        *,
        max_rounds: int,
        max_fanout: int,
    ) -> None:
        _nonnegative_integer(max_rounds, "max_rounds")
        _nonnegative_integer(max_fanout, "max_fanout")
        if set(agents) != set(graph.agent_ids):
            raise ValueError("agents must exactly match graph agent IDs")
        if any(agent.agent_id != agent_id for agent_id, agent in agents.items()):
            raise ValueError("agent mapping keys must match runtime agent IDs")
        self.graph = graph
        self.agents = MappingProxyType(dict(agents))
        self.max_rounds = int(max_rounds)
        self.max_fanout = int(max_fanout)

    def run(
        self,
        *,
        episode_id: str,
        source_id: int,
        query: object,
        handler: RoundHandler,
        stop_rule: StopRule | None = None,
    ) -> EpisodeResult:
        if source_id not in self.agents:
            raise ValueError(f"invalid source ID: {source_id!r}")
        if not episode_id:
            raise ValueError("episode_id must not be empty")

        states = {
            agent_id: agent.reset_episode(episode_id, query)
            for agent_id, agent in self.agents.items()
        }
        states[source_id].active = True
        events: list[EpisodeEvent] = []
        sent_message_ids: set[str] = set()

        for round_index in range(self.max_rounds):
            active_agents: list[int] = []
            for agent_id in self.graph.agent_ids:
                state = states[agent_id]
                for message in state.begin_round():
                    events.append(
                        EpisodeEvent(
                            round_index=round_index,
                            event_type="delivered",
                            agent_id=agent_id,
                            message_id=message.message_id,
                            receiver_id=agent_id,
                            wire_bytes=message.cost.wire_bytes,
                        )
                    )
                if state.active:
                    active_agents.append(agent_id)

            for agent_id in active_agents:
                state = states[agent_id]
                events.append(EpisodeEvent(round_index=round_index, event_type="activated", agent_id=agent_id))
                outgoing = tuple(handler(self.agents[agent_id], state, round_index))
                self._send_batch(
                    outgoing,
                    episode_id=episode_id,
                    round_index=round_index,
                    sender_id=agent_id,
                    states=states,
                    sent_message_ids=sent_message_ids,
                    events=events,
                )

            if stop_rule is not None and stop_rule(self.agents[source_id], states[source_id], round_index):
                events.append(EpisodeEvent(round_index=round_index, event_type="stopped", agent_id=source_id))
                return EpisodeResult(episode_id, source_id, "stopped", tuple(events))

        return EpisodeResult(episode_id, source_id, "round_limit", tuple(events))

    def _send_batch(
        self,
        outgoing: tuple[MessageEnvelope, ...],
        *,
        episode_id: str,
        round_index: int,
        sender_id: int,
        states: Mapping[int, AgentEpisodeState],
        sent_message_ids: set[str],
        events: list[EpisodeEvent],
    ) -> None:
        unique_recipients: set[int] = set()
        valid_outgoing: list[MessageEnvelope] = []
        for message in outgoing:
            self._validate_send(message, episode_id, round_index, sender_id)
            if message.message_id in sent_message_ids:
                events.append(
                    EpisodeEvent(round_index, "duplicate", sender_id, message.message_id, message.receiver_id)
                )
                continue
            unique_recipients.add(message.receiver_id)
            valid_outgoing.append(message)
        if len(unique_recipients) > self.max_fanout:
            raise ValueError("per-round fan-out exceeds max_fanout")

        for message in valid_outgoing:
            sent_message_ids.add(message.message_id)
            delivered = replace(message, ttl=message.ttl - 1)
            accepted = states[message.receiver_id].queue_for_next_round(delivered)
            events.append(
                EpisodeEvent(
                    round_index=round_index,
                    event_type="sent" if accepted else "duplicate",
                    agent_id=sender_id,
                    message_id=message.message_id,
                    receiver_id=message.receiver_id,
                    wire_bytes=message.cost.wire_bytes,
                )
            )

    def _validate_send(
        self,
        message: MessageEnvelope,
        episode_id: str,
        round_index: int,
        sender_id: int,
    ) -> None:
        if message.episode_id != episode_id:
            raise ValueError("message belongs to another episode")
        if message.round_sent != round_index:
            raise ValueError("message round_sent does not match current round")
        if message.sender_id != sender_id:
            raise ValueError("message sender does not match active agent")
        if message.ttl <= 0:
            raise ValueError("message TTL is exhausted")
        if not self.graph.has_edge(sender_id, message.receiver_id):
            raise ValueError("message receiver is not a graph neighbor")


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)
