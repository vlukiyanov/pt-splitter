from ptsplitter.utils import (
    embedding_groups,
    iter_get_scores,
    positive_edges,
    negative_edges
)

import networkx as nx
import numpy as np

example_graph = nx.from_edgelist([
    ('1', '2'),
    ('1', '3'),
    ('2', '3'),
    ('6', '4'),
    ('6', '5'),
    ('4', '5'),
    ('3', '6')
])


def test_embedding_groups():
    items = [0, 0, 1, 1, 2]
    groups = embedding_groups(items * 2, [np.array(item) for item in items] * 2)
    assert len(groups) == 3
    assert len(groups[0]) == 4
    assert len(groups[1]) == 4
    assert len(groups[2]) == 2
    assert groups[0][0] == 0
    assert groups[1][0] == 1
    assert groups[2][0] == 2


def test_iter_get_scores():
    items = [0, 0, 1, 1, 2]
    groups = embedding_groups(items * 2, [np.array(item) for item in items] * 2)
    scores01 = list(iter_get_scores(groups, 0, 1))
    assert len(scores01) == 16
    scores12 = list(iter_get_scores(groups, 1, 2))
    assert len(scores12) == 8
    for item in scores01:
        assert item == 0
    for item in scores12:
        assert item == 1


def test_positive_edges():
    edges = list(positive_edges(example_graph))
    actual = set(map(frozenset, [
        ('1', '2'),
        ('1', '3'),
        ('2', '3'),
        ('6', '4'),
        ('6', '5'),
        ('4', '5')
    ]))
    assert len(edges) == 6
    assert set(map(frozenset, edges)) == actual


def test_negative_edges():
    edges = list(negative_edges(example_graph))
    print(edges)
    actual = set(map(frozenset, [
        ('1', '5'),
        ('2', '5'),
        ('3', '5'),
        ('6', '1'),
        ('6', '2'),
        ('4', '1'),
        ('4', '2'),
        ('4', '3')
    ]))
    assert len(edges) == 8  # 8 + 7 = 15
    assert set(map(frozenset, edges)) == actual
