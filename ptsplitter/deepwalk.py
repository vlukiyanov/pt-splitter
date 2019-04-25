from functools import partial, lru_cache
from multiprocessing import cpu_count
import random
from typing import Hashable, Iterator, List, Callable, Dict, Tuple, Iterable, Optional

from cytoolz.itertoolz import take, iterate, sliding_window, partition, mapcat
from gensim.models import Word2Vec
import networkx as nx
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

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


def initial_persona_embedding(Gp: nx.Graph, initial_embedding: Dict[Hashable, np.ndarray]):
    return {persona_node: initial_embedding[persona_node.node] for persona_node in Gp.nodes()}


def to_embedding_matrix(node_embeddings: Dict[Hashable, np.ndarray],
                        embedding_dimension: int,
                        reverse_lookup: Dict[int, Hashable]) -> np.ndarray:
    initial_embedding = np.ndarray((len(node_embeddings), embedding_dimension))
    for index in reverse_lookup:
        initial_embedding[index, :] = node_embeddings[reverse_lookup[index]]
    return initial_embedding


def iter_skip_window_walk(walk: List[Hashable], window_size: int) -> Iterator[Tuple[int, int]]:
    for window in sliding_window(2 * window_size + 1, walk):
        for target in window[:window_size] + window[window_size + 1:]:
            yield (window[window_size], target)


def iter_training_batches(walks: Iterator[List[Hashable]],
                          window_size: int,
                          batch_size: int,
                          forward_lookup_persona: Dict[Hashable, int],
                          forward_lookup: Dict[Hashable, int]) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    for window_batch in partition(batch_size, mapcat(partial(iter_skip_window_walk, window_size=window_size), walks)):
        persona_batch = torch.Tensor(batch_size).long()
        pure_node_batch = torch.Tensor(batch_size).long()
        context_node_batch = torch.Tensor(batch_size).long()
        for index, (source, target) in enumerate(window_batch):
            persona_batch[index] = forward_lookup_persona[source]
            pure_node_batch[index] = forward_lookup[source.node]
            context_node_batch[index] = forward_lookup_persona[target]
        yield persona_batch, pure_node_batch, context_node_batch


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
        self.dataset_size = dataset_size if dataset_size is not None else graph.number_of_nodes() * walk_length
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
