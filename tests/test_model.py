import networkx as nx
from torch.optim import SGD

from ptsplitter.deepwalk import lookup_tables, DeepWalkDataset
from ptsplitter.model import predict, train
from ptsplitter.persona import persona_graph
from ptsplitter.splitter import SplitterEmbedding


def test_train():
    # this just tests that the model training script runs on a real example without throwing errors
    karate = nx.karate_club_graph()
    persona_karate = persona_graph(karate)
    forward_persona, reverse_persona = lookup_tables(persona_karate)
    forward, reverse = lookup_tables(karate)
    embedding = SplitterEmbedding(
        node_count=karate.number_of_nodes(),
        persona_node_count=persona_karate.number_of_nodes(),
        embedding_dimension=100
    )
    optimizer = SGD(embedding.parameters(), lr=0.01)
    dataset = DeepWalkDataset(
        persona_karate,
        3,
        20,
        forward_persona,
        forward
    )
    train(
        dataset=dataset,
        model=embedding,
        epochs=1,
        batch_size=100,
        optimizer=optimizer,
        cuda=False
    )
    assert embedding.persona_embedding.weight.shape == (persona_karate.number_of_nodes(), 100)
    assert embedding.embedding.weight.shape == (karate.number_of_nodes(), 100)


def test_predict():
    karate = nx.karate_club_graph()
    persona_karate = persona_graph(karate)
    forward_persona, reverse_persona = lookup_tables(persona_karate)
    embedding = SplitterEmbedding(
        node_count=karate.number_of_nodes(),
        persona_node_count=persona_karate.number_of_nodes(),
        embedding_dimension=100
    )
    df = predict(reverse_persona, embedding)
    assert len(df) == len(reverse_persona)
    assert set(df.columns) == {'persona_node', 'node', 'index', 'embedding_vector'}
    assert set(df['node']) == set(karate.nodes)
    assert set(df['persona_node']) == set(persona_karate.nodes)
    assert set(df['index']).issuperset({0, 1})

