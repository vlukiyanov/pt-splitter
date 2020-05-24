import operator
from functools import partial

from cytoolz.curried import nth
from cytoolz.functoolz import excepts
from cytoolz.itertoolz import groupby, take
import networkx as nx
from networkx.algorithms.link_prediction import jaccard_coefficient, adamic_adar_index
from sklearn.metrics import roc_auc_score

from ptsplitter.deepwalk import lookup_tables
from ptsplitter.persona import persona_graph
from ptsplitter.utils import positive_edges, negative_edges, iter_get_scores_networkx


print("Reading in dataset.")
G = max(
    nx.connected_component_subgraphs(nx.read_edgelist("data_input/CA-AstroPh.txt")),
    key=len,
)
sample_number = G.number_of_edges() // 2
G_original = nx.Graph(G)
positive_samples = list(take(sample_number, positive_edges(G)))
negative_samples = list(take(sample_number, negative_edges(G)))
G.remove_edges_from(positive_samples)

positive_scores_non_persona = list(
    map(nth(2), jaccard_coefficient(G, positive_samples))
)
negative_scores_non_persona = list(
    map(nth(2), jaccard_coefficient(G, negative_samples))
)

print(sum(positive_scores_non_persona))
print(sum(negative_scores_non_persona))

print(
    roc_auc_score(
        [1] * len(positive_samples) + [0] * len(negative_samples),
        positive_scores_non_persona + negative_scores_non_persona,
    )
)

print("Constructing persona graph.")
PG = persona_graph(G_original)

print("Constructing lookups.")
forward_persona, reverse_persona = lookup_tables(PG)
forward, reverse = lookup_tables(G)

groups = groupby(operator.attrgetter("node"), PG.nodes())

positive_scores_persona = [
    excepts(ValueError, max, lambda _: 0.0)(
        iter_get_scores_networkx(groups, node1, node2, PG, jaccard_coefficient)
    )
    for (node1, node2) in positive_samples
]
negative_scores_persona = [
    excepts(ValueError, max, lambda _: 0.0)(
        iter_get_scores_networkx(groups, node1, node2, PG, jaccard_coefficient)
    )
    for (node1, node2) in negative_samples
]
print(sum(positive_scores_persona))
print(sum(negative_scores_persona))

print(
    roc_auc_score(
        [1] * len(positive_samples) + [0] * len(negative_samples),
        positive_scores_persona + negative_scores_persona,
    )
)
