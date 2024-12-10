from provnet_utils import *
from collections import defaultdict,deque

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
    edge_src = [edge[0] for edge in original_G.edges(data=True)]
    edge_dst = [edge[1] for edge in original_G.edges(data=True)]
    timestamps = [edge[2]["time"] for edge in original_G.edges(data=True)]

    adj_list = defaultdict(list)
    for src, dst, t in zip(edge_src[0], edge_dst[1], timestamps):
        adj_list[src].append((dst, t))

    for (src, dst) in new_G.edges():
        new_G[src][dst]["time"] = find_earliest_paths(adj_list, src, dst)

    return new_G

def add_timestamps_to_graph(original_G: nx.Graph, new_G: nx.Graph) -> nx.Graph:
    for (src, dst) in new_G.edges():
        timestamps = [edge[2]["time"] for edge in original_G.edges(data=True) if edge[1] == dst]
        earliest = min(timestamps)
        new_G[src][dst]["time"] = earliest

    return new_G


def find_earliest_paths(adj_list, s, d):
    """
    Find the earliest paths for multiple (s, d) pairs.

    :param adj_list: Preprocessed adjacency list with timestamps.
    :param sd_pairs: List of (s, d) pairs to compute earliest paths for.
    :return: A dictionary mapping (s, d) pairs to (earliest_path, earliest_time).
    """
    results = {}

    queue = deque([(s, [], 0)])  # (current_node, path, max_timestamp_so_far)
    earliest_time = float('inf')
    best_path = None
    visited = {}  # Cache the earliest time to reach each node

    while queue:
        current_node, path, current_time = queue.popleft()
        new_path = path + [current_node]

        if current_node == d:
            if current_time < earliest_time:
                earliest_time = current_time
                best_path = new_path
            continue

        for neighbor, edge_time in adj_list[current_node]:
            new_time = max(current_time, edge_time)
            if neighbor not in visited or new_time < visited[neighbor]:
                visited[neighbor] = new_time
                queue.append((neighbor, new_path, new_time))

    return earliest_time