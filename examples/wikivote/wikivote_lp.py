from cytoolz.curried import nth
from cytoolz.itertoolz import take
import networkx as nx
from networkx.algorithms.link_prediction import jaccard_coefficient, adamic_adar_index
from sklearn.metrics import roc_auc_score

from ptsplitter.deepwalk import (
    lookup_tables
)
from ptsplitter.persona import persona_graph
from ptsplitter.utils import (
    positive_edges,
    negative_edges
)


print('Reading in dataset.')
G = nx.read_edgelist('data_input/wiki-Vote.txt')
sample_number = G.number_of_edges() // 2
G_original = nx.Graph(G)
positive_samples = list(take(sample_number, positive_edges(G)))
negative_samples = list(take(sample_number, negative_edges(G)))
G.remove_edges_from(positive_samples)

# print('Constructing persona graph.')
# PG = persona_graph(G)
#
# print('Constructing lookups.')
# forward_persona, reverse_persona = lookup_tables(PG)
# forward, reverse = lookup_tables(G)

positive_scores_non_persona = list(map(nth(2), jaccard_coefficient(G, positive_samples)))
negative_scores_non_persona = list(map(nth(2), jaccard_coefficient(G, negative_samples)))

print(sum(positive_scores_non_persona))
print(sum(negative_scores_non_persona))

print(roc_auc_score(
    [1] * len(positive_samples) + [0] * len(negative_samples),
    positive_scores_non_persona + negative_scores_non_persona
))
