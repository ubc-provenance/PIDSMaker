from tqdm import tqdm
import networkx as nx
import igraph as ig
from provnet_utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

class DEPIMPACT():
    def __init__(self, graph):
        self.graph = graph

        self.forward_adj = self._gen_forward_adj_dict()
        self.backward_adj = self._get_backward_adj_dict()
        self.degree_scores = self._cal_degree_score()

    def gen_dependency_graph(self, poi):
        # print(f"Generating dependency graph starting from node {poi}")
        subgraph_nodes = set()
        if len(self.backward_adj[poi].keys()) > 0:
            entry2path = backward_tracing(poi, self.backward_adj)
            entry2info = {}
            for entry, pathset in entry2path.items():
                # if entry == poi:
                if False:
                    continue
                else:
                    entry2info[entry] = {}

                    unique_nodes = set()
                    for edge in list(pathset):
                        unique_nodes.add(edge[0])
                        unique_nodes.add(edge[1])
                    entry2info[entry]['nodes'] = unique_nodes

                    unique_nodes.discard(poi)
                    node_scores = []
                    for node in unique_nodes:
                        node_scores.append(self.degree_scores[node])
                    if len(node_scores) == 0:
                        entry2info[entry]['score'] = 0
                    else:
                        entry2info[entry]['score'] = sum(node_scores) / len(node_scores)
            entry_scores = []
            for e, info in entry2info.items():
                entry_scores.append((e, info['score']))
            max_entry_score = max(entry_scores, key=lambda x: x[1])[1]
            highest_entries = [item for item in entry_scores if item[1] == max_entry_score]

            # print("Entries with the highest score are: ", highest_entries)

            for entry, score in highest_entries:
                subgraph_nodes |= entry2info[entry]['nodes']
        else:
            # print(f'No backward subgraph for poi node {poi}')
            pass

        if len(self.forward_adj[poi].keys()) > 0:
            exit2path = forward_tracing(poi, self.forward_adj)
            exit2info = {}
            for exit, pathset in exit2path.items():
                # if exit == poi:
                if False:
                    continue
                else:
                    exit2info[exit] = {}

                    unique_nodes = set()
                    for edge in list(pathset):
                        unique_nodes.add(edge[0])
                        unique_nodes.add(edge[1])
                    exit2info[exit]['nodes'] = unique_nodes

                    unique_nodes.discard(poi)
                    node_scores = []
                    for node in unique_nodes:
                        node_scores.append(self.degree_scores[node])
                    if len(node_scores) == 0:
                        exit2info[exit]['score'] = 0
                    else:
                        exit2info[exit]['score'] = sum(node_scores) / len(node_scores)
            exit_scores = []
            for e, info in exit2info.items():
                exit_scores.append((e, info['score']))
            max_exit_score = max(exit_scores, key=lambda x: x[1])[1]
            highest_exits = [item for item in exit_scores if item[1] == max_exit_score]

            # print("Exits with the highest score are: ", highest_exits)

            for exit, score in highest_exits:
                subgraph_nodes |= exit2info[exit]['nodes']
        else:
            # print(f'No forward subgraph for poi node {poi}')
            pass

        subgraph_nodes.add(poi)

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
    visual_style["layout"] = G.layout("kk")

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


