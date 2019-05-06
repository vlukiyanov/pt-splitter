from collections import Counter
from cytoolz.itertoolz import take
import networkx as nx

from ptsplitter.deepwalk import iter_random_walk, iter_random_walks, lookup_tables, initial_deepwalk_embedding, \
    to_embedding_matrix, iter_skip_window_walk


graph_abcd = nx.from_edgelist([
    ('a', 'b'),
    ('b', 'c'),
    ('c', 'd')
])

graph_ab = nx.from_edgelist([('a', 'b')])

graph_abc_weighted = nx.from_edgelist([('a', 'b'), ('a', 'c'), ('b', 'c')])
graph_abc_weighted['a']['b']['weight'] = 1
graph_abc_weighted['a']['c']['weight'] = 2

# TODO some of the tests below are not ideally deterministic, replace with deterministic


def test_basic_iter_random_walk():
    assert set(take(10 ** 5, iter_random_walk(graph_abcd, 'a'))) == {'a', 'b', 'c', 'd'}
    assert set(take(2, iter_random_walk(graph_ab, 'a'))) == {'a', 'b'}
    assert set(take(2, iter_random_walk(graph_ab, 'b'))) == {'a', 'b'}


def test_basic_iter_random_walk_weighted():
    assert set(take(10 ** 5, iter_random_walk(graph_abc_weighted, 'a', weight='weight'))) == {'a', 'b', 'c'}
    count = Counter(take(10 ** 5, iter_random_walk(graph_abc_weighted, 'a', weight='weight')))
    assert count['c'] > count['b']


def test_basic_iter_random_walks():
    for walk in map(set, (take(10, iter_random_walks(graph_ab, 2)))):
        assert walk == {'a', 'b'}


def test_basic_lookup_tables():
    forward, reverse = lookup_tables(graph_ab)
    assert set(forward.keys()) == {'a', 'b'}
    assert set(reverse.values()) == {'a', 'b'}
    assert set(forward.values()) == {0, 1}
    assert set(reverse.keys()) == {0, 1}


def test_basic_initial_deepwalk_embedding():
    forward, reverse = lookup_tables(graph_ab)
    walks = take(100, iter_random_walks(graph_ab, 2))
    embedding = initial_deepwalk_embedding(walks, forward, 10)
    assert len(embedding) == 2
    assert set(embedding.keys()) == {'a', 'b'}


def test_to_embedding_matrix():
    forward, reverse = lookup_tables(graph_ab)
    walks = take(100, iter_random_walks(graph_ab, 2))
    node_embedding = initial_deepwalk_embedding(walks, forward, 10)
    embedding = to_embedding_matrix(node_embedding, 10, reverse)
    # TODO add more thorough tests that check what this is actually doing
    assert embedding.shape == (2, 10)


def test_iter_skip_window_walk():
    walk = [0, 1, 2, 3]  # 1 and 2 are the center nodes for window_size 1
    expected = {
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3)
    }
    assert expected == set(iter_skip_window_walk(walk, 1))
