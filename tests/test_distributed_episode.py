import pytest

from medlatent.distributed import (
    AgentRuntime,
    MessageEnvelope,
    MessageKind,
    RequestPayload,
    SynchronousEpisodeEngine,
    SyntheticKnowledgeRecord,
    SyntheticKnowledgeStore,
    complete_graph,
    path_graph,
)


def _agent(agent_id: int, term: str) -> AgentRuntime:
    return AgentRuntime(
        agent_id,
        SyntheticKnowledgeStore((SyntheticKnowledgeRecord(f"r-{agent_id}", (term,), f"DX:{agent_id}"),)),
    )


def _request(message_id: str, round_sent: int, sender_id: int, receiver_id: int, ttl: int = 1):
    return MessageEnvelope.create(
        message_id=message_id,
        episode_id="episode-1",
        round_sent=round_sent,
        sender_id=sender_id,
        receiver_id=receiver_id,
        kind=MessageKind.REQUEST,
        ttl=ttl,
        payload=RequestPayload("consult"),
    )


def test_agent_retrieves_only_its_own_private_store():
    first = _agent(0, "alpha")
    second = _agent(1, "beta")

    assert [record.record_id for record in first.retrieve_local("alpha")] == ["r-0"]
    assert second.retrieve_local("alpha") == ()
    assert not hasattr(first, "store")
    assert not hasattr(second, "records")


def test_episode_state_isolated_between_episodes_and_deduplicates_messages():
    agent = _agent(1, "beta")
    first = agent.start_episode("episode-1", "alpha")
    message = _request("m-1", 0, 0, 1)

    assert first.queue_for_next_round(message)
    assert not first.queue_for_next_round(message)
    assert first.begin_round() == (message,)
    second = agent.reset_episode("episode-2", "beta")

    assert second is not first
    assert second.current_inbox == ()
    assert second.next_inbox == []
    assert second.received_messages == []
    assert second.seen_message_ids == set()


def test_engine_delivers_messages_in_the_next_round_and_is_deterministic():
    agents = {0: _agent(0, "alpha"), 1: _agent(1, "beta")}
    engine = SynchronousEpisodeEngine(path_graph(2), agents, max_rounds=2, max_fanout=1)
    observations: list[tuple[int, int, tuple[str, ...]]] = []

    def handler(agent, state, round_index):
        observations.append((round_index, agent.agent_id, tuple(message.message_id for message in state.current_inbox)))
        if agent.agent_id == 0 and round_index == 0:
            return (_request("m-1", round_index, 0, 1),)
        state.deactivate()
        return ()

    first = engine.run(episode_id="episode-1", source_id=0, query="alpha", handler=handler)
    second = engine.run(episode_id="episode-1", source_id=0, query="alpha", handler=handler)

    assert (0, 0, ()) in observations
    assert (1, 1, ("m-1",)) in observations
    assert first.events == second.events
    assert first.termination_reason == "round_limit"


def test_engine_rejects_illegal_edges_and_fanout_violations():
    path_agents = {0: _agent(0, "a"), 1: _agent(1, "b"), 2: _agent(2, "c")}
    path_engine = SynchronousEpisodeEngine(path_graph(3), path_agents, max_rounds=1, max_fanout=1)

    with pytest.raises(ValueError, match="not a graph neighbor"):
        path_engine.run(
            episode_id="episode-1",
            source_id=0,
            query="a",
            handler=lambda agent, state, round_index: (_request("m-illegal", 0, 0, 2),),
        )

    complete_agents = {0: _agent(0, "a"), 1: _agent(1, "b"), 2: _agent(2, "c")}
    complete_engine = SynchronousEpisodeEngine(complete_graph(3), complete_agents, max_rounds=1, max_fanout=1)
    with pytest.raises(ValueError, match="fan-out"):
        complete_engine.run(
            episode_id="episode-1",
            source_id=0,
            query="a",
            handler=lambda agent, state, round_index: (
                _request("m-1", 0, 0, 1),
                _request("m-2", 0, 0, 2),
            ),
        )


def test_engine_rejects_expired_ttl_and_applies_source_stop_rule():
    agents = {0: _agent(0, "a"), 1: _agent(1, "b")}
    engine = SynchronousEpisodeEngine(path_graph(2), agents, max_rounds=2, max_fanout=1)

    with pytest.raises(ValueError, match="TTL is exhausted"):
        engine.run(
            episode_id="episode-1",
            source_id=0,
            query="a",
            handler=lambda agent, state, round_index: (
                _request("expired", round_index, 0, 1, ttl=0),
            ),
        )

    result = engine.run(
        episode_id="episode-2",
        source_id=0,
        query="a",
        handler=lambda agent, state, round_index: (),
        stop_rule=lambda agent, state, round_index: round_index == 0,
    )

    assert result.termination_reason == "stopped"
    assert result.events[-1].event_type == "stopped"
