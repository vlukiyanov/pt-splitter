from itertools import product
from typing import Any, Callable, Dict, Iterable, List, TypeVar
import numpy as np

from cytoolz.curried import groupby, valmap
from cytoolz.itertoolz import getter
from cytoolz.functoolz import thread_first

T = TypeVar('T')


def embedding_groups(node_list: List[T], persona_embedding_list: List[np.ndarray]) -> Dict[T, List[np.ndarray]]:
    """
    Utility function, which given aligned list of nodes and embedding lists from the model.predict function,
    obtain a dictionary from base graph nodes to a list of embeddings. The order of the embeddings for the
    base nodes is not ordered, and the order may differ on different calls.

    :param node_list: list of base nodes, which is duplicated
    :param persona_embedding_list: corresponding embeddings
    :return: dictionary mapping base nodes to all their embeddings
    """
    return thread_first(
        zip(node_list, persona_embedding_list),
        groupby(getter(0)),
        valmap(lambda x: list(map(getter(1), x)))
    )


def iter_get_scores(groups: Dict[T, List[np.ndarray]],
                    node1: T,
                    node2: T,
                    product_function: Callable[[np.ndarray, np.ndarray], Any] = np.dot) -> Iterable[float]:
    """
    Iterate all scores between two nodes in the base graph by looking at embeddings of all of their
    personas. You can then apply some function to this like max, min or mean.

    :param groups: lookup from base node to all their embeddings, output of embedding_groups
    :param node1: first node, must be present as a key in groups
    :param node2: second node, must be present as a key in groups
    :param product_function: function to use to compute the product of the two embeddings, default np.dot
    :return: iterator of product_function applied to all possible pairs of embeddings
    """
    return (
        float(product_function(embedding1, embedding1))
        for embedding1, embedding2 in product(groups[node1], groups[node2])
    )
