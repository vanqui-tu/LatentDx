import numpy as np
import pytest

from medlatent.distributed import (
    CommunicationGraph,
    complete_graph,
    erdos_renyi_graph,
    path_graph,
    ring_graph,
    stochastic_block_model_graph,
    watts_strogatz_graph,
)


def test_graph_rejects_non_simple_or_asymmetric_adjacency():
    with pytest.raises(ValueError, match="symmetric"):
        CommunicationGraph(np.array([[0, 1], [0, 0]]))
    with pytest.raises(ValueError, match="self-loops"):
        CommunicationGraph(np.array([[1, 0], [0, 0]]))


def test_m2_graph_helpers_and_paths_are_deterministic():
    assert complete_graph(3).edge_list() == ((0, 1), (0, 2), (1, 2))
    assert path_graph(4).shortest_path_length(0, 3) == 3
    assert ring_graph(5).neighbors(0) == (1, 4)
    assert ring_graph(5).shortest_path_length(0, 2) == 2


def test_optional_graph_generators_are_seeded_and_return_current_graph_type():
    pytest.importorskip("networkx")
    erdos_first = erdos_renyi_graph(8, 0.35, seed=42)
    erdos_second = erdos_renyi_graph(8, 0.35, seed=42)
    watts_first = watts_strogatz_graph(8, 2, 0.4, seed=42)
    watts_second = watts_strogatz_graph(8, 2, 0.4, seed=42)
    sbm_first = stochastic_block_model_graph([4, 4], [[0.8, 0.2], [0.2, 0.7]], seed=42)
    sbm_second = stochastic_block_model_graph([4, 4], [[0.8, 0.2], [0.2, 0.7]], seed=42)

    assert isinstance(erdos_first, CommunicationGraph)
    assert np.array_equal(erdos_first.adjacency, erdos_second.adjacency)
    assert np.array_equal(watts_first.adjacency, watts_second.adjacency)
    assert np.array_equal(sbm_first.adjacency, sbm_second.adjacency)


def test_optional_graph_connectivity_policy_uses_current_graph_api():
    pytest.importorskip("networkx")
    graph = erdos_renyi_graph(3, 0.0, seed=42, connectivity="allow_disconnected")

    assert graph.shortest_path_length(0, 1) is None
    with pytest.raises(ValueError, match="connectivity='connected'"):
        erdos_renyi_graph(3, 0.0, seed=42, connectivity="connected")
