import networkx as nx

def main(G: nx.Graph) -> nx.Graph:
    """
    Removes edges that make cycles in the graph, including self-loops.
    Also removes duplicate edges.
    Basically makes graph G a DAG.
    """
    G = nx.DiGraph(G) # removes duplicate edges
    G.remove_edges_from(nx.selfloop_edges(G))
    
    cycles = nx.find_cycle(G)
    for u, v in cycles:
        G.remove_edge(u, v)

    return G
