from config import *
from provnet_utils import *

from .transformation_utils import add_arbitrary_timestamps_to_graph

def identify_root_nodes(G):
    root_nodes = set()

    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        in_edges = list(G.in_edges(node, data=True))

        # out edge timestamp
        earliest_out_time = min(edge[2]['time'] for edge in out_edges) if out_edges else None
        # in
        earliest_in_time = min(edge[2]['time'] for edge in in_edges) if in_edges else None

        # if root
        if earliest_out_time is not None and (earliest_in_time is None or earliest_out_time < earliest_in_time):
            root_nodes.add(node)

    return root_nodes

def create_pseudo_graph(G,root_nodes):
    """
    Create a pseudo-graph G' based on the original graph G.
    Each pseudo-root retains the initial feature vector of the original root node,
    and outgoing edges are added from pseudo-roots to their descendants.

    Args:
        G (nx.DiGraph): Original directed graph with nodes and features.

    Returns:
        nx.DiGraph: Pseudo-graph with pseudo-root nodes and directed edges to descendants.
    """
    pseudo_graph = nx.DiGraph()

    # Step 1: Add all original nodes and edges to the pseudo-graph
    for node, attr in G.nodes(data=True):
        pseudo_graph.add_node(node, **attr)


    # Step 3: Create pseudo-root nodes and add edges to descendants
    for root in root_nodes:
        # Create pseudo-root node (retaining the same initial feature vector)
        pseudo_root = f"pseudo_{root}"
        pseudo_graph.add_node(pseudo_root, **G.nodes[root])  # Copy features from root node

        # Add edges from pseudo-root to all descendants of the original root
        descendants = nx.descendants(G, root)
        for descendant in descendants:
            pseudo_graph.add_edge(pseudo_root, descendant)

    return pseudo_graph

def prune_pseudo_roots(pseudo_graph, G, prune_threshold):
    """
    Prune pseudo-root nodes from the pseudo-graph if they connect to more than
    a certain percentage of nodes in the original provenance graph.

    Args:
        pseudo_graph (nx.DiGraph): The pseudo-graph with pseudo-root nodes.
        G (nx.DiGraph): The original provenance graph.
        prune_threshold (float): The threshold as a percentage (0-1) of total nodes in G.
                                 If a pseudo-root connects to more than this percentage of nodes,
                                 it will be pruned.

    Returns:
        nx.DiGraph: The pruned pseudo-graph.
    """
    total_nodes_in_G = len(G.nodes())
    max_allowed_connections = prune_threshold * total_nodes_in_G

    # Identify pseudo-roots that need to be pruned
    pseudo_roots_to_prune = []
    for node in pseudo_graph.nodes():
        if node.startswith("pseudo_"):
            # Count the number of nodes this pseudo-root connects to
            num_connections = len(list(pseudo_graph.successors(node)))
            if num_connections > max_allowed_connections:
                pseudo_roots_to_prune.append(node)

    # Prune the identified pseudo-root nodes from the pseudo-graph
    for pseudo_root in pseudo_roots_to_prune:
        pseudo_graph.remove_node(pseudo_root)

    return pseudo_graph

def remove_pseudo_prefix(graph):
    # Rename nodes
    mapping = {node: node.replace("pseudo_", "") for node in graph.nodes()}
    graph = nx.relabel_nodes(graph, mapping)

    # Rename edges
    for u, v, data in graph.edges(data=True):
        # Update edge attributes if they contain "pseudo_"
        for key in list(data.keys()):
            if isinstance(data[key], str) and "pseudo_" in data[key]:
                data[key] = data[key].replace("pseudo_", "")
    return graph

def main(graph: nx.Graph, cfg) -> nx.Graph:
    root_nodes = identify_root_nodes(graph)
    pseudo_graph = create_pseudo_graph(graph, root_nodes)
    
    use_pruning = cfg.preprocessing.transformation.rcaid_pseudo_graph.use_pruning
    if use_pruning:
        pseudo_graph = prune_pseudo_roots(pseudo_graph, graph, 0.5)
        
    pseudo_graph = remove_pseudo_prefix(pseudo_graph)
    pseudo_graph = add_arbitrary_timestamps_to_graph(original_G=graph, new_G=pseudo_graph)

    return pseudo_graph
