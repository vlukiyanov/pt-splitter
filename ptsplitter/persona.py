from collections import namedtuple
from typing import Callable, Iterable, Sequence, Hashable

import networkx as nx

PersonaNode = namedtuple('PersonaNode', 'node, index')


def create_personas(G: nx.Graph,
                    n: Hashable,
                    clustering: Callable[[nx.Graph], Iterable[Sequence[Hashable]]] = nx.connected_components):
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
                  clustering: Callable[[nx.Graph], Iterable[Sequence[Hashable]]] = nx.connected_components):
    edge_remap = {}
    for n in G.nodes():
        _, persona_remap = create_personas(G, n, clustering)
        edge_remap[n] = persona_remap
    persona_graph_edges = [
        (edge_remap[edge[0]][edge[1]], edge_remap[edge[1]][edge[0]])
        for edge in G.edges()
    ]
    return nx.from_edgelist(persona_graph_edges)
