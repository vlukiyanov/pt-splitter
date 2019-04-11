from cytoolz.itertoolz import take
import networkx as nx

from ptsplitter.deepwalk import iter_random_walk, iter_random_walks, lookup_tables, initial_deepwalk_embedding


graph_abcd = nx.from_edgelist([
    ('a', 'b'),
    ('b', 'c'),
    ('c', 'd')
])

graph_ab = nx.from_edgelist([('a', 'b')])


def test_basic_iter_random_walk():
    assert set(take(10 ** 5, iter_random_walk(graph_abcd, 'a'))) == {'a', 'b', 'c', 'd'}
    assert set(take(2, iter_random_walk(graph_ab, 'a'))) == {'a', 'b'}
    assert set(take(2, iter_random_walk(graph_ab, 'b'))) == {'a', 'b'}


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
