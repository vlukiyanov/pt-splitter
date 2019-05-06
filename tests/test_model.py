import networkx as nx
from torch.optim import SGD
from unittest.mock import Mock

from ptsplitter.deepwalk import lookup_tables, PersonaDeepWalkDataset
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
    dataset = PersonaDeepWalkDataset(
        graph=persona_karate,
        window_size=3,
        walk_length=20,
        dataset_size=10,
        forward_lookup_persona=forward_persona,
        forward_lookup=forward
    )
    scheduler = Mock()
    epoch_callback = Mock()
    train(
        dataset=dataset,
        model=embedding,
        scheduler=scheduler,
        epochs=1,
        batch_size=100,
        optimizer=optimizer,
        epoch_callback=epoch_callback,
        cuda=False
    )
    assert embedding.persona_embedding.weight.shape == (persona_karate.number_of_nodes(), 100)
    assert embedding.embedding.weight.shape == (karate.number_of_nodes(), 100)
    assert scheduler.step.call_count == 1
    assert epoch_callback.call_count == 1


def test_predict():
    karate = nx.karate_club_graph()
    persona_karate = persona_graph(karate)
    forward_persona, reverse_persona = lookup_tables(persona_karate)
    embedding = SplitterEmbedding(
        node_count=karate.number_of_nodes(),
        persona_node_count=persona_karate.number_of_nodes(),
        embedding_dimension=100
    )
    data = predict(reverse_persona, embedding)
    for item in data.keys():
        assert len(data[item]) == len(reverse_persona)
    assert set(data.keys()) == {'persona_node', 'node', 'index', 'embedding_vector'}
    assert set(data['node']) == set(karate.nodes)
    assert set(data['persona_node']) == set(persona_karate.nodes)
    assert set(data['index']).issuperset({0, 1})
