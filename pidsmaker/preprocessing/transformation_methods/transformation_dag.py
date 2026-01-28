"""DAG transformation for provenance graphs.

Converts provenance graphs to Directed Acyclic Graphs (DAGs) by removing cycles
and self-loops. Some PIDS systems assume acyclic structure for efficiency or
simplicity in backward/forward tracing algorithms.
"""

import networkx as nx


def main(G: nx.Graph) -> nx.Graph:
    """
    Removes edges that make cycles in the graph, including self-loops.
    Also removes duplicate edges.
    Basically makes graph G a DAG.
    """
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    while True:
        try:
            cycles = nx.find_cycle(G)
            for cycle in cycles:  # fast approximation, the optimal would be to remove 1 by 1
                G.remove_edge(*cycle)

        except nx.NetworkXNoCycle:
            break
    G = nx.MultiDiGraph(G)
    return G
