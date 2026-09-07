from dataclasses import dataclass

import pytest

from medlatent.distributed import AgentRuntime, MessageEnvelope, MessageKind, RequestPayload, SynchronousEpisodeEngine, complete_graph, path_graph


@dataclass(frozen=True)
class Record:
    label: str


class Store:
    def retrieve(self, query, *, limit=None):
        return (Record(str(query)),)


def _agent(agent_id):
    return AgentRuntime(agent_id, Store())


def _request(round_sent, sender, receiver):
    return MessageEnvelope.create(
        message_id=f"{round_sent}-{sender}-{receiver}", episode_id="e", round_sent=round_sent,
        sender_id=sender, receiver_id=receiver, kind=MessageKind.REQUEST, payload=RequestPayload(),
    )


def test_next_round_delivery_and_private_retrieval_boundary():
    agents = {0: _agent(0), 1: _agent(1)}
    seen = []

    def handler(agent, state, round_index):
        seen.append((round_index, agent.agent_id, tuple(message.message_id for message in state.current_inbox)))
        if agent.agent_id == 0 and round_index == 0:
            assert agent.retrieve_active_episode(state.episode_id)[0].label == "q"
            return (_request(round_index, 0, 1),)
        state.deactivate()
        return ()

    SynchronousEpisodeEngine(path_graph(2), agents, max_rounds=2, max_fanout=1).run(
        episode_id="e", source_id=0, query="q", handler=handler
    )
    assert (1, 1, ("0-0-1",)) in seen
    with pytest.raises(RuntimeError, match="activated"):
        agents[0].retrieve_active_episode("e")


def test_engine_rejects_non_neighbor_and_fanout():
    agents = {index: _agent(index) for index in range(3)}
    with pytest.raises(ValueError, match="not a graph neighbor"):
        SynchronousEpisodeEngine(path_graph(3), agents, max_rounds=1, max_fanout=1).run(
            episode_id="e", source_id=0, query="q", handler=lambda agent, state, round_index: (_request(0, 0, 2),)
        )
    with pytest.raises(ValueError, match="fan-out"):
        SynchronousEpisodeEngine(complete_graph(3), agents, max_rounds=1, max_fanout=1).run(
            episode_id="e", source_id=0, query="q",
            handler=lambda agent, state, round_index: (_request(0, 0, 1), _request(0, 0, 2)),
        )
