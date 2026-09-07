import numpy as np
import pytest

from medlatent.distributed import CommunicationGraph, complete_graph, path_graph, ring_graph


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
