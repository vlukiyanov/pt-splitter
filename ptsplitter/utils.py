from itertools import product
import random
from typing import Any, Callable, Dict, Tuple, Iterable, List, TypeVar

from cytoolz.curried import unique, take, groupby, valmap
from cytoolz.itertoolz import getter
from cytoolz.functoolz import pipe
import networkx as nx
import numpy as np


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
    return pipe(
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


def _iter_positive_edges(G: nx.Graph) -> Iterable[Tuple[Any, Any]]:
    """
    Given a graph, yield edges which are positive samples; these can be removed from the
    graph without disconnecting the graph.

    :param G: input NetworkX graph object
    :return: iterate tuples, each representing an edge
    """
    bridges = list(nx.bridges(G))
    edges = list(G.edges())
    while True:
        choice = random.choice(edges)
        if choice not in bridges:
            yield choice


def _iter_negative_edges(G: nx.Graph) -> Iterable[Tuple[Any, Any]]:
    """
    Given a graph, yield (non) edges which are negative samples; none of these edges should
    be contained in the graph.

    :param G: input NetworkX graph object
    :return: iterate tuples, each representing an edge
    """
    edges = list(G.edges())
    nodes = list(G.nodes())
    while True:
        node1 = random.choice(nodes)
        node2 = random.choice(nodes)
        if node1 != node2 and (node1, node2) not in edges:
            yield (node1, node2)


def _unique_edges(iterable: Iterable[Tuple[Any, Any]], quantity: int) -> List[Tuple[Any, Any]]:
    return list(pipe(iterable, take(quantity), unique))


def positive_edges(G: nx.Graph, quantity: int) -> List[Tuple[Any, Any]]:
    """
    Given a graph, return a unique list of edges which are positive samples; these can
    be removed from the graph without disconnecting the graph.

    :param G: input NetworkX graph object
    :param quantity: maximum number of unique edges
    :return: list of tuples, each representing an edge
    """
    return _unique_edges(_iter_positive_edges(G), quantity)


def negative_edges(G: nx.Graph, quantity: int) -> List[Tuple[Any, Any]]:
    """
    Given a graph, return a unique list of (non) edges which are negative samples; none
    of these edges should be contained in the graph.

    :param G: input NetworkX graph object
    :param quantity: maximum number of unique edges
    :return: list of tuples, each representing an edge
    """
    return _unique_edges(_iter_negative_edges(G), quantity)
