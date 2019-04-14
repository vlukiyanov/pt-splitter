from collections import namedtuple
from typing import Callable, Iterable, Sequence, Hashable, Tuple, List, Dict

import networkx as nx

PersonaNode = namedtuple('PersonaNode', 'node, index')


def create_personas(G: nx.Graph,
                    n: Hashable,
                    clustering: Callable[[nx.Graph], Iterable[Sequence[Hashable]]]
                    ) -> Tuple[List[PersonaNode], Dict[Hashable, PersonaNode]]:
    """
    Given a graph, a node in the graph, and a clustering algorithm, generate the personas for the given node.

    :param G: input graph
    :param n: node in the graph
    :param clustering: algorithm to cluster node on a graph, a callable taking a graph to an iterable of hashable
    sequences
    :return: 2-tuple which holds: personalities, so PersonaNode objects, for the given node; and a remap dictionary
    which maps every node in the ego-net of n in G to its corresponding PersonaNode of n
    """
    clusters = clustering(G.subgraph(G.neighbors(n)))
    persona_remap = {}
    personalities = []
    for index, cluster in enumerate(clusters):
        persona = PersonaNode(node=n, index=index)
        personalities.append(persona)
        for other_node in cluster:
            persona_remap[other_node] = persona
    return personalities, persona_remap


def persona_graph(G: nx.Graph,
                  clustering: Callable[[nx.Graph], Iterable[Sequence[Hashable]]] = nx.connected_components
                  ) -> nx.Graph:
    """
    Construct the persona graph of a graph G, this preserves any edge attributes.

    :param G: input graph
    :param clustering: algorithm to cluster node on a graph, a callable taking a graph to an iterable of hashable
    sequences
    :return: persona graph
    """
    # TODO preserve node data
    edge_remap = {}
    for n in G.nodes():
        _, persona_remap = create_personas(G, n, clustering)
        edge_remap[n] = persona_remap
    persona_graph_edges = [
        (edge_remap[start][end], edge_remap[end][start], data)
        for start, end, data in G.edges(data=True)
    ]
    return nx.from_edgelist(persona_graph_edges)
