from tqdm import tqdm
import networkx as nx
import igraph as ig
from provnet_utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
from datetime import datetime

class DEPIMPACT():
    def __init__(self, graph, poi):
        self.graph = graph
        self.poi = poi
        self.dag, self.dag_poi, self.forward_tags = self.convert_DAG()

        # self.forward_adj = self._gen_forward_adj_dict()
        # self.backward_adj = self._get_backward_adj_dict()
        self.degree_scores = self._cal_degree_score()

    def gen_dependency_graph(self):
        poi = self.dag_poi
        poi_in_graph = self.poi

        subgraph_nodes = set()

        if self.dag.in_degree(poi) > 0:
            # entry2path = dag_backward_tracing(poi, self.dag)
            entry2path = tag_based_backward_tracing(poi, self.dag, self.forward_tags)
            entry2nodes = {}
            for entry, paths in entry2path.items():
                entry_in_graph = entry.split('-')[0]
                if entry_in_graph not in entry2nodes:
                    entry2nodes[entry_in_graph] = set()

                nodes_in_dag = set()
                for path in paths:
                    nodes_in_dag |= set(path)

                for node_in_dag in list(nodes_in_dag):
                    node_in_graph = node_in_dag.split('-')[0]
                    entry2nodes[entry_in_graph].add(node_in_graph)

            entry2score = {}
            for entry, nodes in entry2nodes.items():
                nodes.discard(poi_in_graph)
                node_scores = []
                for node in nodes:
                    node_scores.append(self.degree_scores[node])

                if len(node_scores) == 0:
                    entry2score[entry] = 0
                else:
                    entry2score[entry] = sum(node_scores) / len(node_scores)

            entry_scores = []
            for e, score in entry2score.items():
                entry_scores.append((e, score))
            max_entry_score = max(entry_scores, key=lambda x: x[1])[1]
            highest_entries = [item[0] for item in entry_scores if item[1] == max_entry_score]

            for he in highest_entries:
                subgraph_nodes |= entry2nodes[he]
        else:
            print(f"POI {poi} is an entry node, skip backward tracing.")

        if self.dag.out_degree(poi) > 0:
            exit2path = dag_forward_tracing(poi, self.dag)
            exit2nodes = {}
            for exit, paths in exit2path.items():
                exit_in_graph = exit.split('-')[0]
                if exit_in_graph not in exit2nodes:
                    exit2nodes[exit_in_graph] = set()

                nodes_in_dag = set()
                for path in paths:
                    nodes_in_dag |= set(path)

                for node_in_dag in list(nodes_in_dag):
                    node_in_graph = node_in_dag.split('-')[0]
                    exit2nodes[exit_in_graph].add(node_in_graph)

            exit2score = {}
            for exit, nodes in exit2nodes.items():
                nodes.discard(poi_in_graph)
                node_scores = []
                for node in nodes:
                    node_scores.append(self.degree_scores[node])

                if len(node_scores) == 0:
                    exit2score[exit] = 0
                else:
                    exit2score[exit] = sum(node_scores) / len(node_scores)

            exit_scores = []
            for e, score in exit2score.items():
                exit_scores.append((e, score))
            max_exit_score = max(exit_scores, key=lambda x: x[1])[1]
            highest_entries = [item[0] for item in exit_scores if item[1] == max_exit_score]

            for he in highest_entries:
                subgraph_nodes |= exit2nodes[he]
        else:
            print(f"POI {poi} is an exit node, skip forward tracing.")

        subgraph_nodes.add(poi_in_graph)

        return subgraph_nodes

    def _gen_forward_adj_dict(self):
        forward_adj = {}
        for src, dst, k, attrs in tqdm(self.graph.edges(data=True, keys=True), desc=
                                       'generating forward_adj dictionary'):
            if src not in forward_adj:
                forward_adj[src] = {}
            if dst not in forward_adj:
                forward_adj[dst] = {}
            if dst not in forward_adj[src]:
                forward_adj[src][dst] = []
            forward_adj[src][dst].append(attrs['time'])
        return forward_adj

    def _get_backward_adj_dict(self):
        backward_adj = {}
        for src, dst, k, attrs in tqdm(self.graph.edges(data=True, keys=True),desc=
                                       'generating backward_adj dictionary'):
            if dst not in backward_adj:
                backward_adj[dst] = {}
            if src not in backward_adj:
                backward_adj[src] = {}
            if src not in backward_adj[dst]:
                backward_adj[dst][src] = []
            backward_adj[dst][src].append(attrs['time'])
        return backward_adj

    def _cal_degree_score(self):
        out_to_in = {}
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        for node in tqdm(self.graph.nodes(), desc="calculating degree score"):
            if int(in_degrees[node]) == 0:
                out_to_in[node] = 0
            else:
                out_to_in[node] = int(out_degrees[node]) / int(in_degrees[node])
        return out_to_in

    def convert_DAG(self):
        graph = self.graph
        edges = []
        node_version = {}
        for u, v, k, data in graph.edges(keys=True, data=True):
            edges.append((u, v, int(data['time'])))
            if u not in node_version:
                node_version[u] = 0
            if v not in node_version:
                node_version[v] = 0

        sorted_edges = sorted(edges, key=lambda x: x[2])

        new_nodes = set()
        new_edges = []
        visited = set()

        forward_tags = {}

        for u, v, t in sorted_edges:

            if u == v:
                continue

            src = str(u) + '-' + str(node_version[u])

            if src not in forward_tags:
                forward_tags[src] = set()
            if u not in visited:
                forward_tags[src].add(src)

            visited.add(u)
            new_nodes.add(src)

            if v not in visited:
                dst = str(v) + '-' + str(node_version[v])

                forward_tags[dst] = set()
                forward_tags[dst] |= forward_tags[src]

                visited.add(v)
                new_nodes.add(dst)
                new_edges.append((src, dst, {'time': int(t)}))
            else:
                dst_current = str(v) + '-' + str(node_version[v])
                dst_new = str(v) + '-' + str(node_version[v] + 1)

                forward_tags[dst_new] = set()
                forward_tags[dst_new] |= forward_tags[src]
                forward_tags[dst_new] |= forward_tags[dst_current]

                node_version[v] += 1
                new_nodes.add(dst_new)
                new_edges.append((src, dst_new, {'time': int(t)}))
                new_edges.append((dst_current, dst_new, {'time': int(t)}))

        DAG = nx.DiGraph()
        DAG.add_nodes_from(list(new_nodes))
        DAG.add_edges_from(new_edges)

        old_poi = self.poi
        new_poi = str(old_poi) + '-' + str(node_version[old_poi])

        return DAG, new_poi, forward_tags

def backward_tracing(poi: str, backward_adj: dict):
    queue = [(poi, float('inf'),[])]
    entry2path = {}
    # visited = set()

    while queue:
        current_node, current_time, path = queue.pop(0)

        # if current_node in visited:
        #     continue
        # visited.add(current_node)

        is_entry_node = True
        for predecessor in backward_adj[current_node].keys():
            edge_time = find_max_smaller_than(backward_adj[current_node][predecessor], current_time)
            if edge_time is not None:
                is_entry_node = False
                new_path = path + [(predecessor, current_node)]
                queue.append((predecessor, edge_time, new_path))

        if is_entry_node:
            if current_node not in entry2path:
                entry2path[current_node] = set()
            entry2path[current_node] |= set(path)

    return entry2path

def forward_tracing(poi: str, forward_adj: dict):
    queue = [(poi, -float('inf'),[])]
    exit2path = {}

    while queue:
        current_node, current_time, path = queue.pop(0)

        is_exit_node = True
        for successor in forward_adj[current_node].keys():
            edge_time = find_min_larger_than(forward_adj[current_node][successor], current_time)
            if edge_time is not None:
                is_exit_node = False
                new_path = path + [(current_node, successor)]
                queue.append((successor, edge_time, new_path))

        if is_exit_node:
            if current_node not in exit2path:
                exit2path[current_node] = set()
            exit2path[current_node] |= set(path)

    return exit2path

def dag_backward_tracing(poi: str, dag: nx.DiGraph):
    entries = [n for n in dag.nodes() if dag.in_degree(n) == 0]
    entry2path = {}
    for e in entries:
        # all_paths = list(nx.all_simple_paths(dag, e, poi))
        try:
            shortest_path = nx.shortest_path(dag, e, poi)
            all_paths = [shortest_path]
        except:
            all_paths = [[]]
        entry2path[e] = all_paths
    return entry2path

def dag_forward_tracing(poi: str, dag: nx.DiGraph):
    exits = [n for n in dag.nodes() if dag.out_degree(n) == 0]
    exit2path = {}
    for e in exits:
        # all_paths = list(nx.all_simple_paths(dag, poi, e))
        try:
            shortest_path = nx.shortest_path(dag, e, poi)
            all_paths = [shortest_path]
        except:
            all_paths = [[]]
        exit2path[e] = all_paths
    return exit2path

def tag_based_backward_tracing(poi: str, dag: nx.DiGraph, tags: dict):
    log_with_pid("start tag based backward tracing")
    visited = set()
    stack = [poi]
    entry2nodes = {}
    while stack:
        current_node = stack.pop()

        for entry in tags[current_node]:
            if entry not in entry2nodes:
                entry2nodes[entry] = set()
            entry2nodes[entry].add(current_node)

        visited.add(current_node)

        for parent in dag.predecessors(current_node):
            if parent not in visited:
                stack.append(parent)

    result = {}
    for entry_node, node_set in entry2nodes.items():
        result[entry_node] = [list(node_set)]

    return result


def find_min_larger_than(sequence, value):
    min_larger = None
    for num in sequence:
        if num > value:
            if min_larger is None or num < min_larger:
                min_larger = num
    return min_larger

def find_max_smaller_than(sequence, value):
    max_smaller = None
    for num in sequence:
        if num < value:
            if max_smaller is None or num > max_smaller:
                max_smaller = num
    return max_smaller

def log_with_pid(msg: str, *args):
    pid = os.getpid()
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - (pid: {pid}) - {msg}", *args)

def visualize_dependency_graph(dependency_graph,
                               ground_truth_nids,
                               poi,
                               tw,
                               out_dir,
                               cfg):
    node_to_path_type = get_node_to_path_and_type(cfg) #key type is int

    edge_index = np.array([(int(u), int(v)) for u, v, k, attrs in dependency_graph.edges(data=True, keys=True)])
    unique_nodes, new_edge_index = np.unique(edge_index.flatten(), return_inverse=True)
    new_edge_index = new_edge_index.reshape(edge_index.shape)
    unique_paths = [f'{str(n)}:'+node_to_path_type[int(n)]['path'] for n in unique_nodes]
    unique_types = [node_to_path_type[int(n)]["type"] for n in unique_nodes]
    unique_labels = [int(n) in ground_truth_nids for n in unique_nodes]

    G = ig.Graph(edges=[tuple(e) for e in new_edge_index], directed=True)
    G.vs["original_id"] = unique_nodes
    G.vs["path"] = unique_paths
    G.vs["type"] = unique_types
    G.vs["shape"] = ["rectangle" if typ == "file" else "circle" if typ == "subject" else "triangle" for typ in
                     unique_types]

    G.vs["label"] = unique_labels

    BENIGN = "#44BC"
    ATTACK = "#FF7E79"
    POI = "red"
    TRACED = "green"

    visual_style = {}
    visual_style["bbox"] = (700, 700)
    visual_style["margin"] = 40
    visual_style["layout"] = G.layout("kk", maxiter=100)

    visual_style["vertex_size"] = 13
    visual_style["vertex_width"] = 13
    visual_style["vertex_label_dist"] = 1.3
    visual_style["vertex_label_size"] = 6
    visual_style["vertex_label_font"] = 1
    visual_style["vertex_color"] = [ATTACK if label else BENIGN for label in G.vs["label"]]
    visual_style["vertex_label"] = G.vs["path"]
    visual_style["vertex_frame_width"] = 1
    visual_style["vertex_frame_color"] = [POI if int(n) in poi else TRACED for n in unique_nodes]

    visual_style["edge_curved"] = 0.1
    visual_style["edge_width"] = 1 #[3 if label else 1 for label in y_hat]
    visual_style["edge_color"] = "gray" # ["red" if label else "gray" for label in subgraph.es["y"]]
    visual_style["edge_arrow_size"] = 8
    visual_style["edge_arrow_width"] = 8

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the graph using igraph
    plot = ig.plot(G, target=ax, **visual_style)

    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=BENIGN, label='Benign/FP'),
        mpatches.Patch(color=ATTACK, label='Attack/TP'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10, label='Subject', markeredgewidth=1),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=10, label='File', markeredgewidth=1),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=10, label='IP', markeredgewidth=1),
        mpatches.Patch(edgecolor=TRACED, label='traced node', facecolor='none'),
        mpatches.Patch(edgecolor=POI, label='POI node', facecolor='none')
    ]

    # Add legend to the plot
    ax.legend(handles=legend_handles, loc='upper right', fontsize='medium')

    # Save the plot with legend
    out_file = f"attack_graph_in_tw_{tw}"
    svg = os.path.join(out_dir, f"{out_file}.png")
    plt.savefig(svg)
    plt.close(fig)

    log(f"Figure saved to {svg}")


