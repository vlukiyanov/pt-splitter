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
        ego_node_count=15,
        embedding_dimension=10
    )
    batch = torch.ones(5).long()
    persona_batch = torch.ones(5).long()
    output = embedding(batch, persona_batch)
    assert output['embedding'].shape == (5, 10)
    assert output['persona_embedding'].shape == (5, 10)


def test_embedding_graph():
    forward, reverse = lookup_tables(graph_abcd)
    walks = take(10, iter_random_walks(graph_abcd, 5))
    node_embeddings = initial_deepwalk_embedding(walks, forward, 10)
    initial_embedding = to_embedding_matrix(node_embeddings, 10, reverse)
    embedding = SplitterEmbedding(
        node_count=10,
        ego_node_count=15,
        embedding_dimension=10,
        initial_embedding=initial_embedding
    )
    batch = torch.ones(5).long()
    persona_batch = torch.ones(5).long()
    output = embedding(batch, persona_batch)
    assert output['embedding'].shape == (5, 10)
    assert output['persona_embedding'].shape == (5, 10)
