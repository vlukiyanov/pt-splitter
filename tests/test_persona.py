import networkx as nx

from ptsplitter.persona import create_personas, persona_graph

graph_figure = nx.from_edgelist(
    [
        ("a", "b"),
        ("b", "c"),
        ("c", "a"),
        ("c", "e"),
        ("c", "d"),
        ("c", "f"),
        ("e", "f"),
        ("d", "f"),
        ("f", "h"),
        ("f", "g"),
        ("h", "g"),
    ]
)

graph_ab_weight = nx.from_edgelist(
    [("a", "b", {"weight": 1}), ("b", "c", {"weight": 1})]
)


def test_create_personas():
    personas_a = create_personas(graph_figure, "a", nx.connected_components)
    assert len(personas_a[0]) == 1
    personas_c = create_personas(graph_figure, "c", nx.connected_components)
    assert len(personas_c[0]) == 2
    personas_f = create_personas(graph_figure, "f", nx.connected_components)
    assert len(personas_f[0]) == 2


def test_persona_graph():
    Gp = persona_graph(graph_figure)
    assert Gp.number_of_edges() == 11
    assert Gp.number_of_nodes() == 10


def test_persona_graph_data():
    Gp = persona_graph(graph_ab_weight)
    assert Gp.number_of_edges() == 2
    for _, _, data in Gp.edges(data=True):
        assert data["weight"] == 1
