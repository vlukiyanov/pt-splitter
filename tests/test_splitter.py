from cytoolz.itertoolz import take
import networkx as nx
import torch

from ptsplitter.deepwalk import to_embedding_matrix, iter_random_walks, lookup_tables, initial_deepwalk_embedding
from ptsplitter.splitter import SplitterEmbedding


graph_abcd = nx.from_edgelist([
    ('a', 'b'),
    ('b', 'c'),
    ('c', 'd')
])


def test_embedding_basic():
    embedding = SplitterEmbedding(
        node_count=10,
        persona_node_count=15,
        embedding_dimension=100
    )
    persona_batch = torch.ones(5).long()
    output = embedding(persona_batch)
    assert output.shape == (5, 100)


def test_embedding_graph():
    forward, reverse = lookup_tables(graph_abcd)
    walks = take(10, iter_random_walks(graph_abcd, 5))
    node_embeddings = initial_deepwalk_embedding(walks, forward, 100)
    initial_embedding = to_embedding_matrix(node_embeddings, 100, reverse)
    embedding = SplitterEmbedding(
        node_count=10,
        persona_node_count=15,
        embedding_dimension=100,
        initial_embedding=initial_embedding
    )
    persona_batch = torch.ones(5).long()
    output = embedding(persona_batch)
    assert output.shape == (5, 100)


def test_loss_basic():
    embedding = SplitterEmbedding(
        node_count=10,
        persona_node_count=15,
        embedding_dimension=100,
    )
    persona_batch = torch.ones(5).long()
    pure_node_batch = torch.tensor([0, 1, 2, 0, 1])
    context_node_batch = torch.tensor([0, 2, 1, 4, 2])
    output = embedding.loss(persona_batch, pure_node_batch, context_node_batch)
    assert output.shape == tuple()
    output.backward()
