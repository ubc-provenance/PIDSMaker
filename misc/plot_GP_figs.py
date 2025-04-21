import csv
import os

import decendant_of_attack_root as source
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import threatrace_ground_truth as neigh
import torch
from tqdm import tqdm

from pidsmaker.config import *
from pidsmaker.utils.labelling import get_uuid2nids
from pidsmaker.utils.utils import *
from pidsmaker.utils.utils import datetime_to_ns_time_US

dataset_to_mtw = {
    "THEIA_E3": [
        [91, 92, 93, 94],
        [6, 7, 8],
    ],
}


def gen_GP_graph(cfg, start_time, end_time, nids, node_dir, edge_dir):
    malicious_nodes = list(nids)  # list(int)
    cur, connect = init_database_connection(cfg)

    node_to_path_type = get_node_to_path_and_type(cfg)  # key type is int

    log("Get edges between GPs")
    rows = source.get_events_between_GPs(cur, start_time, end_time, malicious_nodes)
    edge_set = set()
    for row in tqdm(rows):
        src_id = row[1]
        operation = row[2]
        dst_id = row[4]

        if operation in rel2id:
            edge_set.add((src_id, dst_id, operation))

    edge_list = list(edge_set)

    edge_index = np.array([(int(u), int(v)) for u, v, _ in edge_list])
    unique_nodes, new_edge_index = np.unique(edge_index.flatten(), return_inverse=True)
    new_edge_index = new_edge_index.reshape(edge_index.shape)
    unique_paths = [node_to_path_type[int(n)]["path"] for n in unique_nodes]
    unique_types = [node_to_path_type[int(n)]["type"] for n in unique_nodes]

    G = ig.Graph(edges=[tuple(e) for e in new_edge_index], directed=True)
    G.vs["original_id"] = unique_nodes
    G.vs["path"] = unique_paths
    G.vs["type"] = unique_types
    G.vs["shape"] = [
        "rectangle" if typ == "file" else "circle" if typ == "subject" else "triangle"
        for typ in unique_types
    ]

    node_list = unique_nodes.tolist()
    new_node_list = list(range(len(node_list)))
    node_label = [1 if n in malicious_nodes else 0 for n in node_list]
    edge_list = new_edge_index.tolist()
    log(f"There are {len(unique_nodes)} nodes")

    with open(node_dir, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])
        writer.writerows(zip(new_node_list, node_label))
    log(f"node list is saved to {node_dir}")

    with open(edge_dir, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target"])
        writer.writerows(edge_list)
    log(f"edge list is saved to {edge_dir}")

    return G


def main(cfg, attack_num, gt_type, plot_gt):
    cur, connect = init_database_connection(cfg)
    log(f"Start processing attack {attack_num} of dataset {cfg.dataset.name}")

    base_dir = os.path.join(ROOT_ARTIFACT_DIR, "ground_truth_figs/")
    os.makedirs(base_dir, exist_ok=True)

    out_dir = os.path.join(base_dir, f"{cfg.dataset.name}_attack_{attack_num}/")
    os.makedirs(out_dir, exist_ok=True)

    log("get orthrus GPs")
    orthrus_nids = get_GP_of_one_attack(cfg, attack_num)  # set(int)

    log("generate nx graph for GPs")
    _, start_time, end_time = cfg.dataset.attack_to_time_window[attack_num]

    if gt_type == "orthrus":
        node_save_path = os.path.join(out_dir, "orthrus_nodes.csv")
        edge_save_path = os.path.join(out_dir, "orthrus_edges.csv")
        start_timestamp = datetime_to_ns_time_US(start_time)
        end_timestamp = datetime_to_ns_time_US(end_time)

        ig_graph = gen_GP_graph(
            cfg,
            start_timestamp,
            end_timestamp,
            orthrus_nids,
            node_dir=node_save_path,
            edge_dir=edge_save_path,
        )

        log("start visualization")
        out_path = os.path.join(out_dir, "orthrus_ground_truth.svg")

        if plot_gt:
            visualize_ig_graph(G=ig_graph, out_path=out_path)

    elif gt_type == "neigh":
        node_save_path = os.path.join(out_dir, "neigh_nodes.csv")
        edge_save_path = os.path.join(out_dir, "neigh_edges.csv")

        # start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        # end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        #
        # start_of_day = start_dt.replace(hour=0, minute=0, second=0)
        # end_of_day = end_dt.replace(hour=23, minute=59, second=59)
        #
        # start_of_day_str = start_of_day.strftime('%Y-%m-%d %H:%M:%S')
        # end_of_day_str = end_of_day.strftime('%Y-%m-%d %H:%M:%S')

        day_graph = neigh.gen_graph(
            datetime_to_ns_time_US(start_time), datetime_to_ns_time_US(end_time), cfg
        )
        n = 2
        log(f"Get {n}-hop neighbors of GPs")
        GPs = [str(nid) for nid in orthrus_nids]
        new_gps = neigh.get_n_hop_of_GP(day_graph, GPs, n)
        log(f"there are {len(new_gps)} neigh GPs")

        neigh_nids = set()
        for gp in new_gps:
            neigh_nids.add(int(gp))

        start_timestamp = datetime_to_ns_time_US(start_time)
        end_timestamp = datetime_to_ns_time_US(end_time)

        ig_graph = gen_GP_graph(
            cfg,
            start_timestamp,
            end_timestamp,
            neigh_nids,
            node_dir=node_save_path,
            edge_dir=edge_save_path,
        )

        log("start visualization")
        out_path = os.path.join(out_dir, "neigh_ground_truth.svg")

        if plot_gt:
            visualize_ig_graph(G=ig_graph, out_path=out_path)

    elif gt_type == "batch":
        node_save_path = os.path.join(out_dir, "batch_nodes.csv")
        edge_save_path = os.path.join(out_dir, "batch_edges.csv")

        ig_graph = batch_type_graph(
            cfg,
            orthrus_nids,
            attack_num=attack_num,
            node_dir=node_save_path,
            edge_dir=edge_save_path,
        )

        log("start visualization")
        out_path = os.path.join(out_dir, "batch_ground_truth.svg")
        if plot_gt:
            visualize_ig_graph(G=ig_graph, out_path=out_path)

    elif gt_type == "source":
        node_save_path = os.path.join(out_dir, "source_nodes.csv")
        edge_save_path = os.path.join(out_dir, "source_edges.csv")
        start_timestamp = datetime_to_ns_time_US(start_time)
        end_timestamp = datetime_to_ns_time_US(end_time)

        log("Get events between GPs")
        rows = source.get_events_between_GPs(
            cur, start_timestamp, end_timestamp, list(orthrus_nids)
        )

        edges = []
        for row in rows:
            src_id = row[1]
            operation = row[2]
            dst_id = row[4]
            t = row[6]
            if operation in rel2id:
                edges.append((str(src_id), str(dst_id), int(t)))

        dag_between_GPs, _ = source.generate_DAG(edges)
        log("Get root nodes")
        root_nodes = set(
            [node for node, in_degree in dag_between_GPs.in_degree() if in_degree == 0]
        )
        log(f"Root nodes are: {root_nodes}")

        log("Get events in attack time range")
        rows = source.get_events_between_time_range(cur, start_timestamp, end_timestamp)
        edges = []
        for row in rows:
            src_id = row[1]
            operation = row[2]
            dst_id = row[4]
            t = row[6]
            if operation in rel2id:
                edges.append((str(src_id), str(dst_id), int(t)))

        dag_of_attack, node_version = source.generate_DAG(edges)

        all_descendants = set()
        for root in root_nodes:
            descendants = nx.descendants(dag_of_attack, root)
            desc = set([v.split("-")[0] for v in descendants])
            all_descendants |= desc

        log(f"{len(all_descendants)} descedants of root nodes in the attack")

        source_gps = [int(n) for n in all_descendants]
        ig_graph = gen_GP_graph(
            cfg,
            start_timestamp,
            end_timestamp,
            set(source_gps),
            node_dir=node_save_path,
            edge_dir=edge_save_path,
        )

        log("start visualization")
        out_path = os.path.join(out_dir, "source_ground_truth.svg")

        if plot_gt:
            visualize_ig_graph(G=ig_graph, out_path=out_path)


def batch_type_graph(cfg, GPs, attack_num, node_dir, edge_dir):
    dataset_name = cfg.dataset.name

    node_to_path_type = get_node_to_path_and_type(cfg)  # key type is int

    graph_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, cfg.dataset.test_files)

    edge_set = set()
    for tw in dataset_to_mtw[dataset_name][attack_num]:
        log(f"processing tw {tw}")
        nx_graph = torch.load(sorted_paths[tw])
        for u, v, k, data in nx_graph.edges(data=True, keys=True):
            edge_set.add((int(u), int(v)))

    edge_list = list(edge_set)

    edge_index = np.array([(int(u), int(v)) for u, v in edge_list])
    unique_nodes, new_edge_index = np.unique(edge_index.flatten(), return_inverse=True)
    new_edge_index = new_edge_index.reshape(edge_index.shape)
    unique_paths = [node_to_path_type[int(n)]["path"] for n in unique_nodes]
    unique_types = [node_to_path_type[int(n)]["type"] for n in unique_nodes]

    G = ig.Graph(edges=[tuple(e) for e in new_edge_index], directed=True)
    G.vs["original_id"] = unique_nodes
    G.vs["path"] = unique_paths
    G.vs["type"] = unique_types
    G.vs["shape"] = [
        "rectangle" if typ == "file" else "circle" if typ == "subject" else "triangle"
        for typ in unique_types
    ]

    node_list = unique_nodes.tolist()
    new_node_list = list(range(len(node_list)))
    node_label = [1 if n in GPs else 0 for n in node_list]
    edge_list = new_edge_index.tolist()
    log(f"There are {len(unique_nodes)} nodes")

    with open(node_dir, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label"])
        writer.writerows(zip(new_node_list, node_label))
    log(f"node list is saved to {node_dir}")

    with open(edge_dir, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target"])
        writer.writerows(edge_list)
    log(f"edge list is saved to {edge_dir}")

    return G


def visualize_ig_graph(G, out_path, show_text=False):
    node_num = G.vcount()

    BENIGN = "#44BC"
    ATTACK = "#FF7E79"
    POI = "red"
    TRACED = "green"

    visual_style = {}
    visual_style["bbox"] = (700, 700)
    visual_style["margin"] = 10
    visual_style["layout"] = G.layout("kk", maxiter=100)

    visual_style["vertex_size"] = 13
    visual_style["vertex_width"] = 13
    visual_style["vertex_label_dist"] = 1.3
    visual_style["vertex_label_size"] = 6
    visual_style["vertex_label_font"] = 1
    visual_style["vertex_color"] = [BENIGN] * node_num
    if show_text:
        visual_style["vertex_label"] = G.vs["path"]
    visual_style["vertex_frame_width"] = 1
    visual_style["vertex_frame_color"] = [BENIGN] * node_num

    visual_style["edge_curved"] = 0.1
    visual_style["edge_width"] = 1  # [3 if label else 1 for label in y_hat]
    visual_style["edge_color"] = (
        "gray"  # ["red" if label else "gray" for label in subgraph.es["y"]]
    )
    visual_style["edge_arrow_size"] = 8
    visual_style["edge_arrow_width"] = 8

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the graph using igraph
    plot = ig.plot(G, target=ax, **visual_style)

    # Create legend handles
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="k",
            markersize=10,
            label="Subject",
            markeredgewidth=1,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="k",
            markersize=10,
            label="File",
            markeredgewidth=1,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="k",
            markersize=10,
            label="IP",
            markeredgewidth=1,
        ),
    ]

    # Add legend to the plot
    ax.legend(handles=legend_handles, loc="upper right", fontsize="medium")

    # Save the plot with legend
    plt.savefig(out_path)
    plt.close(fig)

    log(f"Figure saved to {out_path}")


def get_GP_of_one_attack(cfg, attack_num):
    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

    gt_file_dir = cfg.dataset.ground_truth_relative_path[attack_num]

    nids = set()
    with open(os.path.join(cfg._ground_truth_dir, gt_file_dir), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            node_uuid, node_labels, _ = row[0], row[1], row[2]
            node_id = uuid2nids[node_uuid]
            nids.add(int(node_id))
    return nids


if __name__ == "__main__":
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    attack_num = args.show_attack
    gt_type = args.gt_type
    plot_gt = args.plot_gt
    cfg = get_yml_cfg(args)

    main(cfg, attack_num, gt_type, plot_gt)
