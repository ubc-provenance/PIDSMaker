import networkx as nx

def main(G: nx.Graph) -> nx.Graph:
    """
    Removes edges that make cycles in the graph, including self-loops.
    Also removes duplicate edges.
    Basically makes graph G a DAG.
    """
    G.remove_edges_from(nx.selfloop_edges(G))
    while True:
        try:
            cycle = nx.find_cycle(G)
            edge_to_remove = cycle[0]
            G.remove_edge(*edge_to_remove)
            
        except nx.NetworkXNoCycle:
            break
    return G
