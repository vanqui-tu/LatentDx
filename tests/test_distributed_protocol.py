import numpy as np
import pytest

from medlatent.distributed import (
    CommunicationGraph,
    DirectNeighborRouter,
    FloodUnvisitedRouter,
    ProposalPayload,
    RandomKRouter,
    RequestPayload,
    StructuredChannel,
    complementary_evidence_fixture,
    majority_vote,
    max_confidence,
    mean_score,
    path_graph,
)


def test_structured_channel_round_trips_and_enforces_byte_limit():
    channel = StructuredChannel(max_wire_bytes=128)
    payload = RequestPayload("consult", ("DX:1",), (0, 2))

    encoded = channel.encode(payload)

    assert channel.decode(encoded) == payload
    assert channel.cost(payload).wire_bytes == len(encoded)
    with pytest.raises(ValueError, match="exceeds"):
        StructuredChannel(max_wire_bytes=4).encode(payload)


def test_fixed_routers_select_only_legal_neighbors_deterministically():
    graph = path_graph(5)

    assert DirectNeighborRouter().select_neighbors(graph, 2, max_fanout=2) == (1, 3)
    assert FloodUnvisitedRouter().select_neighbors(
        graph, 2, max_fanout=2, excluded_agent_ids=(1, 2)
    ) == (3,)
    first = RandomKRouter(seed=42).select_neighbors(graph, 2, max_fanout=1, episode_id="e", round_index=0)
    second = RandomKRouter(seed=42).select_neighbors(graph, 2, max_fanout=1, episode_id="e", round_index=0)
    assert first == second
    assert all(graph.has_edge(2, neighbor) for neighbor in first)


def test_aggregators_are_deterministic():
    proposals = (
        ProposalPayload("DX:A", 0.6, 0.7),
        ProposalPayload("DX:B", 0.9, 0.6),
        ProposalPayload("DX:A", 0.8, 0.9),
    )

    assert majority_vote(proposals).candidate_label == "DX:A"
    assert mean_score(proposals).candidate_label == "DX:B"
    assert max_confidence(proposals).candidate_label == "DX:A"


def test_complementary_fixture_states_private_necessary_agents_and_target():
    fixture = complementary_evidence_fixture()
    source = fixture.build_agents()[fixture.source_id]

    assert fixture.necessary_agent_ids == (1, 2)
    assert fixture.target_label == "DX:combined"
    assert source.retrieve_local(fixture.query) == ()


def test_b0_and_b1_fail_but_b2_succeeds_with_two_hop_complementary_evidence():
    from medlatent.distributed import BaselineKind, run_baseline

    fixture = complementary_evidence_fixture()
    local = run_baseline(fixture, BaselineKind.LOCAL_ONLY, max_rounds=0, max_fanout=2)
    one_hop = run_baseline(fixture, BaselineKind.ONE_HOP, max_rounds=2, max_fanout=2)
    flooding = run_baseline(fixture, BaselineKind.FLOODING, max_rounds=4, max_fanout=2)

    assert local.prediction is None
    assert one_hop.prediction is None
    assert flooding.prediction == fixture.target_label
    assert flooding.contacted_agent_ids == (0, 1, 2)


def test_flooding_requires_request_return_round_budget_and_bridge_edge():
    from medlatent.distributed import BaselineKind, run_baseline

    fixture = complementary_evidence_fixture()
    out_of_budget = run_baseline(fixture, BaselineKind.FLOODING, max_rounds=3, max_fanout=2)
    broken_bridge = complementary_evidence_fixture(
        CommunicationGraph(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))
    )
    disconnected = run_baseline(broken_bridge, BaselineKind.FLOODING, max_rounds=4, max_fanout=2)

    assert out_of_budget.prediction is None
    assert disconnected.prediction is None


def test_flooding_suppresses_duplicate_request_arrivals_and_charges_request_reply():
    from medlatent.distributed import BaselineKind, run_baseline

    fixture = complementary_evidence_fixture(
        CommunicationGraph(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    )
    result = run_baseline(fixture, BaselineKind.FLOODING, max_rounds=4, max_fanout=2)
    events = result.episode.events
    sent = tuple(event for event in events if event.event_type == "sent")

    assert result.prediction == fixture.target_label
    assert any(event.event_type == "duplicate" for event in events)
    assert any(event.wire_bytes > 0 for event in sent)
    assert any("REQUEST" in event.message_id for event in sent)
    assert any("PROPOSAL" in event.message_id for event in sent)
