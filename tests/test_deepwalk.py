from cytoolz.itertoolz import take
import networkx as nx
import pytest

from ptsplitter.deepwalk import iter_random_walk


@pytest.fixture
def graph_abcd():
    return nx.from_edgelist([
        ('a', 'b'),
        ('b', 'c'),
        ('c', 'd')
    ])


@pytest.fixture
def graph_ab():
    return nx.from_edgelist([
        ('a', 'b'),
    ])


def test_basic_iter_random_walk(graph_abcd, graph_ab):
    assert set(take(10 ** 5, iter_random_walk(graph_abcd, 'a'))) == {'a', 'b', 'c', 'd'}
    assert set(take(2, iter_random_walk(graph_ab, 'a'))) == {'a', 'b'}
    assert set(take(2, iter_random_walk(graph_ab, 'b'))) == {'a', 'b'}
