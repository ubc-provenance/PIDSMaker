import contextlib

import networkx as nx
import numpy as np

# Context manager wrapping the draws of random timestamps, set in the processes transforming
# graphs in parallel to draw from the random state that a sequential run would have (see
# `tasks/transformation.py`). None: draws from the current global random state.
random_state_hook = None


def randint_timestamps(min_t, max_t, size):
    """`np.random.randint(min_t, max_t, size=size)` from the global random state."""
    hook = random_state_hook or contextlib.nullcontext
    with hook():
        return np.random.randint(min_t, max_t, size=size)


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
    rand_t = randint_timestamps(min_t, max_t, len(new_G.edges()))

    for t, (src, dst) in zip(rand_t, new_G.edges()):
        new_G[src][dst]["time"] = t

    return new_G
