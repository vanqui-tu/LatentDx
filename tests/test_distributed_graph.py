import numpy as np
import pytest

from medlatent.distributed import (
    CommunicationGraph,
    complete_graph,
    path_graph,
    random_regular_graph,
    ring_graph,
    star_graph,
)


def test_communication_graph_constructs_immutable_undirected_graph():
    source_adjacency = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int8,
    )

    graph = CommunicationGraph(source_adjacency)
    source_adjacency[0, 1] = 0

    assert graph.num_agents == 3
    assert graph.agent_ids == (0, 1, 2)
    assert graph.adjacency.dtype == np.bool_
    assert graph.neighbors(1) == (0, 2)
    assert graph.has_edge(0, 1)
    assert not graph.has_edge(0, 2)
    with pytest.raises(ValueError):
        graph.adjacency[0, 1] = False


def test_communication_graph_rejects_asymmetric_adjacency():
    with pytest.raises(ValueError, match="symmetric"):
        CommunicationGraph(np.array([[0, 1], [0, 0]]))


def test_communication_graph_rejects_self_loops():
    with pytest.raises(ValueError, match="self-loops"):
        CommunicationGraph(np.array([[1, 0], [0, 0]]))


def test_basic_generators_create_expected_undirected_topologies():
    assert complete_graph(4).degree_statistics().minimum == 3
    assert path_graph(4).neighbors(1) == (0, 2)
    assert ring_graph(4).neighbors(0) == (1, 3)
    assert star_graph(4).neighbors(0) == (1, 2, 3)


def test_random_regular_graph_is_seeded_and_regular():
    first = random_regular_graph(8, 3, seed=42)
    second = random_regular_graph(8, 3, seed=42)

    assert np.array_equal(first.adjacency, second.adjacency)
    assert first.degree_statistics().minimum == 3
    assert first.degree_statistics().maximum == 3


@pytest.mark.parametrize(
    ("num_agents", "degree", "message"),
    [
        (5, 3, "must be even"),
        (4, 4, "smaller"),
    ],
)
def test_random_regular_graph_rejects_invalid_degree_combinations(
    num_agents: int, degree: int, message: str
):
    with pytest.raises(ValueError, match=message):
        random_regular_graph(num_agents, degree, seed=42)


def test_path_graph_statistics_and_request_return_feasibility():
    graph = path_graph(4)

    assert graph.connected_components() == ((0, 1, 2, 3),)
    assert graph.shortest_path_length(0, 3) == 3
    assert graph.degree_statistics().minimum == 1
    assert graph.degree_statistics().maximum == 2
    assert graph.degree_statistics().mean == pytest.approx(1.5)
    assert graph.diameter() == 3
    assert graph.minimum_request_return_rounds(0, 3) == 6


def test_ring_and_star_graph_statistics():
    ring = ring_graph(5)
    star = star_graph(4)

    assert ring.shortest_path_length(0, 2) == 2
    assert ring.diameter() == 2
    assert star.connected_components() == ((0, 1, 2, 3),)
    assert star.degree_statistics().minimum == 1
    assert star.degree_statistics().maximum == 3
    assert star.diameter() == 2


def test_disconnected_graph_is_labeled_unreachable():
    graph = CommunicationGraph(np.array([[0, 0], [0, 0]]))

    assert graph.connected_components() == ((0,), (1,))
    assert graph.shortest_path_length(0, 1) is None
    assert graph.minimum_request_return_rounds(0, 1) is None
    with pytest.raises(ValueError, match="disconnected"):
        graph.diameter()
