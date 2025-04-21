import networkx as nx


def main(G: nx.Graph) -> nx.Graph:
    """
    Iterate through each edge in the graph and add the reverse edge if it doesn't exist.
    By default, if an edge (u, v, t) doesn't have a reverse edge (v, u) in the graph,
    an edge (v, u, t) is added. The timestamp t is thus the same for => and <= senses.
    Shouldn't be used with TGN.
    """
    edges_to_add = []
    for u, v, data in G.edges(data=True):
        if not G.has_edge(v, u):
            edges_to_add.append((v, u, data))

    G.add_edges_from(edges_to_add)
    return G
