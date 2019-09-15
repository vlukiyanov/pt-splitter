from cytoolz.itertoolz import take
import networkx as nx
import torch.cuda as cuda
from torch.optim import SGD

from ptsplitter.deepwalk import (
    initial_deepwalk_embedding,
    initial_persona_embedding,
    iter_random_walks,
    lookup_tables,
    to_embedding_matrix,
    PersonaDeepWalkDataset
)
from ptsplitter.model import predict, train
from ptsplitter.persona import persona_graph
from ptsplitter.splitter import SplitterEmbedding
from ptsplitter.utils import (
    embedding_groups,
    positive_edges,
    negative_edges,
    iter_get_scores
)


print('Reading in dataset.')
G = nx.read_edgelist('data_input/wiki-Vote.txt')

print('Constructing persona graph.')
PG = persona_graph(G)

print('Constructing lookups.')
forward_persona, reverse_persona = lookup_tables(PG)
forward, reverse = lookup_tables(G)

print('Generating random walks and initial embeddings.')
walks = take(10000, iter_random_walks(G, length=10))
base_embedding = initial_deepwalk_embedding(
    walks=walks,
    forward_lookup=forward,
    embedding_dimension=100,
    window=10
)
base_matrix = to_embedding_matrix(
    base_embedding,
    embedding_dimension=100,
    reverse_lookup=reverse
)
persona_matrix = to_embedding_matrix(
    initial_persona_embedding(PG, base_embedding),
    embedding_dimension=100,
    reverse_lookup=reverse_persona
)

print('Running splitter.')
print(f'CUDA is{str() if cuda.is_available() else " not"} utilised.')
embedding = SplitterEmbedding(
    node_count=G.number_of_nodes(),
    persona_node_count=PG.number_of_nodes(),
    embedding_dimension=100,
    initial_embedding=base_matrix,
    initial_persona_embedding=persona_matrix,
)

dataset = PersonaDeepWalkDataset(
    graph=PG,
    window_size=3,
    walk_length=50,
    dataset_size=50000,
    forward_lookup_persona=forward_persona,
    forward_lookup=forward
)
if cuda.is_available():
    embedding = embedding.cuda()

optimizer = SGD(embedding.parameters(), lr=0.01)
train(
    dataset=dataset,
    model=embedding,
    epochs=10,
    batch_size=20,
    optimizer=optimizer,
    cuda=cuda.is_available()
)
_, node_list, index_list, persona_embedding_list = predict(reverse_persona, embedding)

groups = embedding_groups(node_list, persona_embedding_list)

negative_samples = take(100, negative_edges(G))
positive_samples = take(100, positive_edges(G))

positive_scores = [max(iter_get_scores(groups, node1, node2)) for (node1, node2) in positive_samples]
negative_scores = [max(iter_get_scores(groups, node1, node2)) for (node1, node2) in negative_samples]

print(1)