import numpy as np
import networkx as nx

def add_arbitrary_timestamps_to_graph(original_G: nx.Graph, new_G: nx.Graph) -> nx.Graph:
    """
    Some transformations change the shape of the graph by adding or removing edges.
    This makes the timestamps associated to edges unusable anymore as we can't associate
    timestamps to newly added edges.
    So we take the min/max timestamp range from the original graph and create random timestamps
    in this range for the new graph.
    Doing this, we can still use the framework, just the order of edges shouldn't be considered.
    For example, TGN shouldn't be used if this function is used previously.
    """
    timestamps = [edge[2]["time"] for edge in original_G.edges(data=True)]
    min_t, max_t = min(timestamps), max(timestamps)
    rand_t = np.random.randint(min_t, max_t, size=len(new_G.edges()))

    for t, (src, dst) in zip(rand_t, new_G.edges()):
        new_G[src][dst]["time"] = t

    return new_G