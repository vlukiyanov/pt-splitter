from ptsplitter.utils import embedding_groups, iter_get_scores

import numpy as np


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
