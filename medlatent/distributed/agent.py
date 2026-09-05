"""Per-agent local runtime and query-scoped episode state."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from .messages import MessageEnvelope, MessageKind, ProposalPayload
from .store import PrivateKnowledgeStore


@dataclass(slots=True)
class AgentEpisodeState:
    agent_id: int
    episode_id: str
    query: object
    current_inbox: tuple[MessageEnvelope, ...] = ()
    next_inbox: list[MessageEnvelope] = field(default_factory=list)
    received_messages: list[MessageEnvelope] = field(default_factory=list)
    seen_message_ids: set[str] = field(default_factory=set)
    local_proposal: ProposalPayload | None = None
    parent_message_id: str | None = None
    parent_agent_id: int | None = None
    active: bool = False

    def queue_for_next_round(self, message: MessageEnvelope) -> bool:
        if message.episode_id != self.episode_id:
            raise ValueError("message belongs to another episode")
        if message.receiver_id != self.agent_id:
            raise ValueError("message receiver does not match state owner")
        if message.message_id in self.seen_message_ids:
            return False
        if message.kind is MessageKind.REQUEST and self.parent_message_id is not None:
            return False
        self.seen_message_ids.add(message.message_id)
        self.next_inbox.append(message)
        if message.kind is MessageKind.REQUEST and self.parent_message_id is None:
            self.parent_message_id = message.message_id
            self.parent_agent_id = message.sender_id
        self.active = True
        return True

    def begin_round(self) -> tuple[MessageEnvelope, ...]:
        self.current_inbox = tuple(self.next_inbox)
        self.next_inbox.clear()
        self.received_messages.extend(self.current_inbox)
        return self.current_inbox

    def set_local_proposal(self, proposal: ProposalPayload | None) -> None:
        self.local_proposal = proposal

    def deactivate(self) -> None:
        self.active = False


class AgentRuntime:
    """An agent with one private store and isolated state per episode."""

    __slots__ = ("_agent_id", "__store", "_states")

    def __init__(self, agent_id: int, store: PrivateKnowledgeStore[Any]) -> None:
        if isinstance(agent_id, bool) or not isinstance(agent_id, Integral) or agent_id < 0:
            raise ValueError("agent_id must be a non-negative integer")
        if not isinstance(store, PrivateKnowledgeStore):
            raise TypeError("store must implement PrivateKnowledgeStore")
        self._agent_id = int(agent_id)
        self.__store = store
        self._states: dict[str, AgentEpisodeState] = {}

    @property
    def agent_id(self) -> int:
        return self._agent_id

    def retrieve_local(self, query: object, *, limit: int | None = None) -> tuple[Any, ...]:
        return self.__store.retrieve(query, limit=limit)

    def start_episode(self, episode_id: str, query: object) -> AgentEpisodeState:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        state = AgentEpisodeState(agent_id=self.agent_id, episode_id=episode_id, query=query)
        self._states[episode_id] = state
        return state

    def state_for(self, episode_id: str) -> AgentEpisodeState:
        try:
            return self._states[episode_id]
        except KeyError as error:
            raise KeyError(f"agent {self.agent_id} has no state for episode {episode_id!r}") from error

    def reset_episode(self, episode_id: str, query: object) -> AgentEpisodeState:
        return self.start_episode(episode_id, query)
