from multiprocessing import cpu_count
import random
from typing import Hashable, Iterator, List, Callable, Dict, Tuple

from cytoolz.itertoolz import take, iterate
from gensim.models import Word2Vec
import networkx as nx
import numpy as np

# TODO figure out iterator versus iterable for typing


def iter_random_walk(G: nx.Graph,
                     n: Hashable,
                     choice: Callable[[List[Hashable]], Hashable] = random.choice
                     ) -> Iterator[Hashable]:
    """
    Given an input graph and a root node, repeatedly yield the results of a random walk starting with the root
    node; if the node is disconnected then the walk will consist just of the node itself.

    :param G: input graph
    :param n: root node
    :param choice: choice function to take a list of nodes and select one randomly
    :return: yields nodes in a random walk, starting with the root node
    """
    if len(G[n]) == 0:
        return
    for cur in iterate(lambda x: choice(list(G[x])), n):
        yield cur


def iter_random_walks(G: nx.Graph,
                      length: int,
                      choice: Callable[[List[Hashable]], Hashable] = random.choice) -> Iterator[List[Hashable]]:
    """
    Given an input graph, repeatedly yield random walks of a fixed maximum length starting at random nodes; if
    the node is disconnected then the walk will consist of the node itself.

    :param G: input graph
    :param length: maximum length of walk
    :param choice: choice function to map a list of nodes to a node
    :return: yields lists of walks
    """
    while True:
        yield list(take(length, iter_random_walk(G, choice(list(G.nodes())), choice)))


def lookup_tables(G: nx.Graph) -> Tuple[Dict[Hashable, int], Dict[int, Hashable]]:
    """
    Given a graph G construct a lookup table between nodes and their integer position in G.nodes() and a reverse
    lookup as dictionaries.

    :param G: input graph
    :return: 2-tuple of node->index and index->node lookups
    """
    forward = {node: index for index, node in enumerate(G.nodes())}
    reverse = {forward[node]: node for node in forward}
    return forward, reverse


def initial_deepwalk_embedding(walks: Iterator[List[int]],
                               forward_lookup: Dict[Hashable, int],
                               embedding_dimension: int,
                               min_count: int = 0,
                               window: int = 10,
                               workers: int = cpu_count()) -> Dict[Hashable, np.ndarray]:
    # TODO constructing the list below might use a lot of memory for larger walks
    model = Word2Vec(
        [[str(forward_lookup[node]) for node in walk] for walk in walks],
        size=embedding_dimension,
        window=window,
        min_count=min_count,
        sg=1,  # use skip-gram
        hs=1,  # use hierarchical softmax
        workers=workers,
        iter=1  # TODO figure out iter setting
    )
    return {node: model.wv[str(forward_lookup[node])] for node in forward_lookup}


def to_embedding_matrix(node_embeddings,
                        embedding_dimension: int,
                        reverse_lookup: Dict[int, Hashable]) -> np.ndarray:
    initial_embedding = np.ndarray((len(node_embeddings), embedding_dimension))
    for index in reverse_lookup:
        initial_embedding[index, :] = node_embeddings[reverse_lookup[index]]
    return initial_embedding
