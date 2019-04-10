import random
from typing import Hashable, Iterator, List, Callable

from cytoolz.itertoolz import take, iterate
import networkx as nx


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
    for cur in iterate(lambda x: choice(list(G[x])), n):
        yield cur
        if len(G[cur]) == 0:
            return


def iter_random_walks(G: nx.Graph,
                      length: int,
                      choice: Callable[[List[Hashable]], Hashable] = random.choice) -> Iterator[List[Hashable]]:
    while True:
        for n in G.nodes():
            yield list(take(length, iter_random_walk(G, n, choice)))
