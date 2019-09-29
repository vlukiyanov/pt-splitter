from itertools import product
import random
from typing import Any, Callable, Dict, Tuple, Iterable, List, Tuple, TypeVar

from cytoolz.curried import groupby, nth, valmap
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
        groupby(0),
        valmap(lambda x: list(map(getter(1), x)))
    )


def positive_edges(G: nx.Graph) -> Iterable[Tuple[Any, Any]]:
    """
    Given a graph, yield edges which are positive samples; these can be removed from the
    graph without disconnecting the graph.

    :param G: input NetworkX graph object
    :return: iterate tuples, each representing an edge
    """
    # TODO this needs more tests
    G = nx.Graph(G)
    edges = list(G.edges())
    random.shuffle(edges)
    for choice in edges:
        G.remove_edge(*choice)
        if nx.is_connected(G):
            yield choice
        else:
            G.add_edge(*choice)


def negative_edges(G: nx.Graph) -> Iterable[Tuple[Any, Any]]:
    """
    Given a graph, yield (non) edges which are negative samples; none of these edges should
    be contained in the graph.

    :param G: input NetworkX graph object
    :return: iterate tuples, each representing an edge
    """
    edges = set(G.edges())
    nodes = list(G.nodes())
    for node1, node2 in product(nodes, nodes):
        if node1 != node2 and (node1, node2) not in edges and (node2, node1) not in edges and hash(node1) < hash(node2):
            yield (node1, node2)


def iter_get_scores(groups: Dict[T, List[np.ndarray]],
                    node1: T,
                    node2: T,
                    product_function: Callable[[np.ndarray, np.ndarray], Any] = np.dot) -> Iterable[float]:
    """
    Iterate all scores between two nodes in the base graph by looking at embeddings of all of their
    personas. You can then apply some function to this like max, min or mean; one minor snag is if
    either of the nodes is not in the lookup dictionary, then this will return an empty iterable, which
    downstream code will have to deal with.

    :param groups: lookup from base node to all their embeddings, output of embedding_groups
    :param node1: first node, must be present as a key in groups
    :param node2: second node, must be present as a key in groups
    :param product_function: function to use to compute the product of the two embeddings, default np.dot
    :return: iterator of product_function applied to all possible pairs of embeddings
    """
    if node1 in groups and node2 in groups:
        yield from (
            float(product_function(embedding1, embedding2))
            for embedding1, embedding2 in product(groups[node1], groups[node2])
        )
    return


def iter_get_scores_networkx(groups: Dict[T, List[np.ndarray]],
                             node1: T,
                             node2: T,
                             G: nx.Graph,
                             product_function: Callable[[nx.Graph, Iterable[Tuple[T, T]]], Iterable[Tuple[T, T, float]]]
                             ) -> Iterable[float]:
    if node1 in groups and node2 in groups:
        yield from map(nth(2), product_function(G, product(groups[node1], groups[node2])))
    return
