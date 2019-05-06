from functools import partial, lru_cache
from multiprocessing import cpu_count
import random
from typing import Hashable, Iterator, List, Dict, Tuple, Optional

from cytoolz.itertoolz import take, iterate, sliding_window, mapcat
from gensim.models import Word2Vec
import networkx as nx
import numpy as np
from torch.utils.data.dataset import Dataset


def iter_random_walk(G: nx.Graph, n: Hashable, weight: Optional[str] = None) -> Iterator[Hashable]:
    """
    Given an input graph and a root node, repeatedly yield the results of a random walk starting with the root
    node; if the node is disconnected then the walk will consist just of the node itself.

    :param G: input graph
    :param n: root node
    :param weight: name of weight attribute to use, or None to disable, default None
    :return: yields nodes in a random walk, starting with the root node
    """
    # TODO this weighted random walk is probably inefficient, using the transition matrix might be better?
    def _next_node(node):
        if len(G[node]) == 1:
            return list(G[node])[0]
        elif weight is None:
            return random.choice(list(G[node]))
        else:
            nodes = []
            weights = []
            for _, to_node, to_weight in G.edges(node, data=weight, default=0):
                nodes.append(to_node)
                weights.append(to_weight)
            weights = np.array(weights)
            return nodes[np.random.choice(np.arange(len(nodes)), p=weights/weights.sum())]
    if len(G[n]) == 0:
        return
    yield from iterate(_next_node, n)


def iter_random_walks(G: nx.Graph, length: int, weight: Optional[str] = None) -> Iterator[List[Hashable]]:
    """
    Given an input graph, repeatedly yield random walks of a fixed maximum length starting at random nodes; if
    the node is disconnected then the walk will consist of the node itself.

    :param G: input graph
    :param length: maximum length of walk
    :param weight: name of weight attribute to use, or None to disable, default None
    :return: yields lists of walks
    """
    while True:
        yield list(take(length, iter_random_walk(G, random.choice(list(G.nodes())), weight=weight)))


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


def to_embedding_matrix(node_embeddings: Dict[Hashable, np.ndarray],
                        embedding_dimension: int,
                        reverse_lookup: Dict[int, Hashable]) -> np.ndarray:
    """
    Given a node embedding lookup, a lookup from index to node, and the embedding dimension (required only to
    construct the array for in-place modification), create the node to embedding numpy array that can then be used
    in the PyTorch network.

    :param node_embeddings: lookup from node to embedding for the graph
    :param embedding_dimension: dimension of the embeddings, which should be constant
    :param reverse_lookup: lookup from integer index to node for the graph
    :return: numpy array of shape [number of nodes, embedding_dimension] filled with the initial embeddings
    """
    initial_embedding = np.ndarray((len(node_embeddings), embedding_dimension))
    for index in reverse_lookup:
        initial_embedding[index, :] = node_embeddings[reverse_lookup[index]]
    return initial_embedding


def iter_skip_window_walk(walk: List[Hashable], window_size: int) -> Iterator[Tuple[int, int]]:
    """
    Given a walk of nodes and a window size, which is interpreted as number of nodes to the left and to the right
    of the node, iteratively yield the central node and a choice of target node from its windows to the left and
    right in the walk.

    :param walk: list of nodes
    :param window_size: number of nodes to the left and to the right
    :return: yields 2-tuples of source and target for training
    """
    for window in sliding_window(2 * window_size + 1, walk):
        for target in window[:window_size] + window[window_size + 1:]:
            yield (window[window_size], target)


def initial_persona_embedding(Gp: nx.Graph, initial_embedding: Dict[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
    """
    Utility function to create the embedding lookup for the personal graph given the embedding lookup for
    the original graph.

    :param Gp: persona graph
    :param initial_embedding: lookup from node to embedding for the original graph
    :return: lookup from node to embedding for the persona graph
    """
    return {persona_node: initial_embedding[persona_node.node] for persona_node in Gp.nodes()}


def initial_deepwalk_embedding(walks: Iterator[List[Hashable]],  # TODO typing
                               forward_lookup: Dict[Hashable, int],
                               embedding_dimension: int,
                               min_count: int = 0,
                               window: int = 10,
                               workers: int = cpu_count()) -> Dict[Hashable, np.ndarray]:
    # TODO constructing the list below might use a lot of memory for larger walks
    # TODO out of vocab issue, if node is not seen in any walk
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


class DeepWalkDataset(Dataset):
    def __init__(self,
                 graph: nx.Graph,
                 window_size: int,
                 walk_length: int,
                 forward_lookup_persona: Dict[Hashable, int],
                 forward_lookup: Dict[Hashable, int],
                 dataset_size: Optional[int] = None) -> None:
        super(DeepWalkDataset, self).__init__()
        self.graph = graph
        self.window_size = window_size
        self.walk_length = walk_length
        self.forward_lookup_persona = forward_lookup_persona
        self.forward_lookup = forward_lookup
        # TODO push this onto caller
        self.dataset_size = dataset_size if dataset_size is not None else graph.number_of_nodes() * walk_length
        # the walker is an infinite iterable that yields new training samples; it is safe to call next on this
        self.walker = mapcat(
            partial(iter_skip_window_walk, window_size=window_size),
            iter_random_walks(graph, walk_length)
        )

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        source, target = next(self.walker)
        return (
            self.forward_lookup_persona[source],
            self.forward_lookup[source.node],
            self.forward_lookup_persona[target],
        )

    def __len__(self):
        return self.dataset_size
