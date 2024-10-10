from collections import defaultdict

from sklearn.metrics import (
    auc,
    roc_curve,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import wandb

from provnet_utils import *
from data_utils import *
from config import *
import igraph as ig
import csv
from sklearn.cluster import KMeans

import labelling
import torch


def get_threshold(val_tw_path, threshold_method: str):
    threshold_method = threshold_method.strip()
    if threshold_method == "max_val_loss":
        return calculate_threshold(val_tw_path)['max']
    elif threshold_method == "mean_val_loss":
        return calculate_threshold(val_tw_path)['mean']
    elif threshold_method == "threatrace":
        return 1.5
    elif threshold_method == "flash":
        return 0.53
    # elif threshold_method == "90_percent_val_loss":
    #     return calculate_threshold(val_tw_path)['percentile_90']
    raise ValueError(f"Invalid threshold method `{threshold_method}`")

def reduce_losses_to_score(losses: list[float], threshold_method: str):
    threshold_method = threshold_method.strip()
    if threshold_method == "mean_val_loss":
        return np.mean(losses)
    elif threshold_method == "max_val_loss":
        return np.max(losses)
    raise ValueError(f"Invalid threshold method {threshold_method}")

def calculate_threshold(val_tw_dir):
    filelist = listdir_sorted(val_tw_dir)

    loss_list = []
    for file in sorted(filelist):
        f = os.path.join(val_tw_dir, file)
        df = pd.read_csv(f).to_dict()
        loss_list.extend(df["loss"].values())

    thr = {
        'max': max(loss_list),
        'mean': mean(loss_list),
        'percentile_90': percentile_90(loss_list)
    }
    log(f"Thresholds: MEAN={thr['mean']:.3f}, STD={std(loss_list):.3f}, MAX={thr['max']:.3f}, 90 Percentile={thr['percentile_90']:.3f}")

    return thr

def calculate_supervised_best_threshold(losses, labels):
    fpr, tpr, thresholds = roc_curve(labels, losses)
    roc_auc = auc(fpr, tpr)

    valid_indices = np.where(tpr >= 0.16)[0]
    fpr_valid = fpr[valid_indices]
    thresholds_valid = thresholds[valid_indices]

    # Find the threshold corresponding to the lowest FPR among valid points
    optimal_idx = np.argmin(fpr_valid)
    optimal_threshold = thresholds_valid[optimal_idx]

    return optimal_threshold

def plot_precision_recall(scores, y_truth, out_file):
    precision, recall, thresholds = precision_recall_curve(y_truth, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    precision_ticks = [i / 20 for i in range(7)]  # Generates [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    plt.yticks(precision_ticks)

    plt.savefig(out_file)

def plot_simple_scores(scores, y_truth, out_file):
    scores_0 = [score for score, label in zip(scores, y_truth) if label == 0]
    scores_1 = [score for score, label in zip(scores, y_truth) if label == 1]

    # Positions on the y-axis for the scatter plot (can be zero or any other constant)
    y_zeros = [0] * len(scores_0)  # All zeros at y=0
    y_ones = [1] * len(scores_1)  # All ones at y=1, you can also keep them at y=0 if you prefer

    plt.figure(figsize=(6, 2))  # Width, height in inches
    plt.scatter(scores_0, y_zeros, color='green')
    plt.scatter(scores_1, y_ones, color='red')

    plt.xlabel('Node anomaly scores')
    plt.yticks([0, 1], ['Benign', 'Malicious'])
    plt.ylim(-0.1, 1.1)  # Adjust if necessary to bring them even closer

    plt.tight_layout()  # Ensures everything fits within the figure area
    plt.savefig(out_file)

def plot_scores_with_paths(scores, y_truth, nodes, max_val_loss_tw, tw_to_malicious_nodes, out_file, cfg):
    node_to_path = get_node_to_path_and_type(cfg)
    paths, types = [], []
    # Prints the path if it exists, else tries to print the cmd line
    for n in nodes:
        types.append(node_to_path[n]["type"])
        path = node_to_path[n]["path"]
        if path == "None":
            paths.append(node_to_path[n]["cmd"] if "cmd" in node_to_path[n] else path)
        else:
            paths.append(path)
            
    # Convert data to numpy arrays for easy manipulation
    scores = np.array(scores)
    y_truth = np.array(y_truth)
    types = np.array(types)

    # Define marker styles for each type
    marker_styles = {
        'subject': 's',   # Square
        'file': 'o',      # Circle
        'netflow': 'D'    # Diamond
    }

    # Separate the scores based on labels
    scores_0 = scores[y_truth == 0]
    scores_1 = scores[y_truth == 1]
    types_0 = types[y_truth == 0]
    types_1 = types[y_truth == 1]
    paths_0 = [path for path, label in zip(paths, y_truth) if label == 0]
    paths_1 = [path for path, label in zip(paths, y_truth) if label == 1]

    plt.figure(figsize=(12, 6))
    
    red = (155/255, 44/255, 37/255)
    green = (62/255, 126/255, 42/255)

    # Plot each type with a different marker for Label 0
    for t in marker_styles.keys():
        plt.scatter(scores_0[types_0 == t], [0]*sum(types_0 == t), 
                    marker=marker_styles[t], color=green, label=f'Label 0 - {t}')

    # Plot each type with a different marker for Label 1
    for t in marker_styles.keys():
        plt.scatter(scores_1[types_1 == t], [1]*sum(types_1 == t), 
                    marker=marker_styles[t], color=red, label=f'Label 1 - {t}')

    # Adding labels and title
    plt.xlabel('Scores')
    plt.ylabel('Labels')
    plt.yticks([0, 1], ['0', '1'])  # Set y-ticks to show label categories
    plt.title('Scatter Plot of Scores by Label')
    plt.legend()

    # Combine scores and paths for easy handling
    combined_scores = list(zip(scores, paths, y_truth, max_val_loss_tw))

    # Sort combined list by scores in descending order
    combined_scores_sorted = sorted(combined_scores, key=lambda x: x[0], reverse=True)

    # Separate the top scores by their labels
    keep_only = 10
    top_0 = [item for item in combined_scores_sorted if item[2] == 0][:keep_only]
    top_1 = [item for item in combined_scores_sorted if item[2] == 1][:keep_only]

    # Annotate the top scores for label 0
    for i, (score, path, _, max_tw_idx) in enumerate(top_0):
        y_position = 0 - (i * 0.1)  # Adjust y-position for each label to avoid overlap
        plt.text(max(scores) + 1, y_position, f"{str(path)[-30:]} ({score:.2f}): TW {max_tw_idx}", fontsize=8, va='center', ha='left', color=green)

    # Annotate the top scores for label 1
    for i, (score, path, _, max_tw_idx) in enumerate(top_1):
        y_position = 1 - (i * 0.1)  # Adjust y-position for each label to avoid overlap and add space between groups
        plt.text(max(scores) + 1, y_position, f"{str(path)[-30:]} ({score:.2f}): TW {max_tw_idx}", fontsize=8, va='center', ha='left', color=red)
        
    plt.text(max(scores) // 3, 1.6, f"Dataset: {cfg.dataset.name}", fontsize=8, va='center', ha='left', color='black')
    plt.text(max(scores) // 3, 1.5, f"Malicious TW: {str(list(tw_to_malicious_nodes.keys()))}", fontsize=8, va='center', ha='left', color='black')

    plt.xlim([min(scores), max(scores) + 7])  # Adjust xlim to make space for text
    plt.ylim([-1, 2])  # Adjust ylim to ensure the text is within the figure bounds
    plt.savefig(out_file)

def plot_false_positives(y_true, y_pred, out_file):
    plt.figure(figsize=(10, 6))
    
    plt.plot(y_pred, label='y_pred', color='blue')
    
    # Adding green dots for true positives (y_true == 1)
    label_indices = [i for i, true in enumerate(y_true) if true == 1]
    plt.scatter(label_indices, [y_pred[i] for i in label_indices], color='green', label='True Positive')
    
    # Adding red dots for false positives (y_true == 0 and y_pred == 1)
    false_positive_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true == 0 and pred == 1]
    plt.scatter(false_positive_indices, [y_pred[i] for i in false_positive_indices], color='red', label='False Positive')
    
    plt.xlabel('Index')
    plt.ylabel('Prediction Value')
    plt.title('True Positives and False Positives in Predictions')
    plt.legend()
    plt.savefig(out_file)

def plot_dor_recall_curve(scores, y_truth, out_file):
    scores = np.array(scores)
    y_truth = np.array(y_truth)
    thresholds = np.linspace(scores.min(), scores.max(), 300)

    sensitivity_list = []
    dor_list = []

    # Iterate over each threshold to calculate recall and DOR
    for threshold in thresholds:
        # Make predictions based on the threshold
        predictions = scores >= threshold
        
        # Calculate TP, FP, TN, FN
        TN, FP, FN, TP = confusion_matrix(y_truth, predictions).ravel()
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # Calculate Diagnostic Odds Ratio (DOR)
        if (FP * FN) == 0:
            dor = np.nan
        else:
            dor = (TP * TN) / (FP * FN)

        sensitivity_list.append(recall)
        dor_list.append(dor)

    # Convert lists to numpy arrays for plotting
    sensitivity_list = np.array(sensitivity_list)
    dor_list = np.array(dor_list)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(sensitivity_list, dor_list, label='DOR vs Sensitivity', color='blue', marker='o')
    plt.xlabel('Sensitivity')
    plt.ylabel('Diagnostic Odds Ratio (DOR)')
    plt.title('Diagnostic Odds Ratio vs Sensitivity at Different Thresholds')
    plt.grid(True)
    plt.legend()
    plt.savefig(out_file)

def get_ground_truth_nids(cfg):
    # ground_truth_nids, ground_truth_paths = [], {}
    # for file in cfg.dataset.ground_truth_relative_path:
    #     with open(os.path.join(cfg._ground_truth_dir, file), 'r') as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             node_uuid, node_labels, node_id = row[0], row[1], row[2]
    #             ground_truth_nids.append(int(node_id))
    #             ground_truth_paths[int(node_id)] = node_labels
    ground_truth_nids, ground_truth_paths, uuid_to_node_id = labelling.get_ground_truth(cfg)
    return set(ground_truth_nids), ground_truth_paths

def get_ground_truth_uuid_to_node_id(cfg):
    # uuid_to_node_id = {}
    # for file in cfg.dataset.ground_truth_relative_path:
    #     with open(os.path.join(cfg._ground_truth_dir, file), 'r') as f:
    #         reader = csv.reader(f)
    #         for row in reader:
    #             node_uuid, node_labels, node_id = row[0], row[1], row[2]
    #             uuid_to_node_id[node_uuid] = node_id
    ground_truth_nids, ground_truth_paths, uuid_to_node_id = labelling.get_ground_truth(cfg)
    return uuid_to_node_id

def get_start_end_from_graph(graph):
    time_list = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        time_list.append(int(data['time']))
    return min(time_list), max(time_list)

def compute_tw_labels(cfg):
    """
    Gets the malcious node IDs present in each time window.
    """
    out_path = cfg.preprocessing.build_graphs._tw_labels
    out_file = os.path.join(out_path, "tw_to_malicious_nodes.pkl")
    uuid_to_node_id = get_ground_truth_uuid_to_node_id(cfg)

    if os.path.exists(out_file):
        os.remove(out_file)
    
    if not os.path.exists(out_file):
        log(f"Computing time-window labels...")
        os.makedirs(out_path, exist_ok=True)

        t_to_node = labelling.get_t2malicious_node(cfg)
        # test_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="test")

        graph_dir = cfg.preprocessing.transformation._graphs_dir
        test_graphs = get_all_files_from_folders(graph_dir, cfg.dataset.test_files)

        num_found_event_labels = 0
        tw_to_malicious_nodes = defaultdict(list)
        for i, tw in enumerate(test_graphs):
            graph = torch.load(tw)
            start, end  = get_start_end_from_graph(graph)

            # start = tw.t.min().item()
            # end = tw.t.max().item()
            
            for t, node_ids in t_to_node.items():
                if start < t < end:
                    for node_id in node_ids: # src, dst, or [src, dst] malicious nodes
                        tw_to_malicious_nodes[i].append(node_id)
                    num_found_event_labels += 1
                    
        log(f"Found {num_found_event_labels}/{len(t_to_node)} edge labels.")
        torch.save(tw_to_malicious_nodes, out_file)
        
    # Used to retrieve node ID from node raw UUID
    # node_labels_path = os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_events_relative_path)

    # uuid_to_node_id = get_ground_truth_uuid_to_node_id(cfg)
    
    # Create a mapping TW number => malicious node IDs
    tw_to_malicious_nodes = torch.load(out_file)
    for tw, nodes in tw_to_malicious_nodes.items():
        unique_nodes, counts = np.unique(nodes, return_counts=True)
        node_to_count = {node: count for node, count in zip(unique_nodes, counts)}
        log(f"TW {tw} -> {len(unique_nodes)} malicious nodes + {len(nodes)} malicious edges")
        
        node_to_count = {uuid_to_node_id[node_id]: count for node_id, count in node_to_count.items()}
        # pprint(node_to_count, width=1)
        tw_to_malicious_nodes[tw] = node_to_count

    return tw_to_malicious_nodes

def datetime_to_ns_time_US(nano_date_str):
    date = nano_date_str.split('.')[0]
    nanos = nano_date_str.split('.')[1]

    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = str(timestamp).split('.')[0] + nanos
    return int(timeStamp)

def compute_tw_labels_for_magic(cfg):
    """
    Gets the malcious node IDs present in each time window.
    """
    out_path = cfg.preprocessing.build_graphs._tw_labels
    out_file = os.path.join(out_path, "tw_to_malicious_nodes.pkl")
    uuid_to_node_id = get_ground_truth_uuid_to_node_id(cfg)

    if os.path.exists(out_file):
        os.remove(out_file)

    if not os.path.exists(out_file):
        log(f"Computing time-window labels...")
        os.makedirs(out_path, exist_ok=True)

        t_to_node = labelling.get_t2malicious_node(cfg)

        base_dir = cfg.preprocessing.build_graphs.magic_graphs_dir
        test_tw = get_all_files_from_folders(base_dir, cfg.dataset.test_files)

        num_found_event_labels = 0
        tw_to_malicious_nodes = defaultdict(list)
        for i, tw in enumerate(test_tw):
            filename = tw.split('/')[-1]
            start_time = filename.split('~')[0]
            end_time = filename.split('~')[1]

            start = datetime_to_ns_time_US(start_time)
            end = datetime_to_ns_time_US(end_time)

            for t, node_ids in t_to_node.items():
                if start < t < end:
                    for node_id in node_ids:  # src, dst, or [src, dst] malicious nodes
                        tw_to_malicious_nodes[i].append(node_id)
                    num_found_event_labels += 1

        log(f"Found {num_found_event_labels}/{len(t_to_node)} edge labels.")
        torch.save(tw_to_malicious_nodes, out_file)

    # Used to retrieve node ID from node raw UUID
    # node_labels_path = os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_events_relative_path)

    # uuid_to_node_id = get_ground_truth_uuid_to_node_id(cfg)

    # Create a mapping TW number => malicious node IDs
    tw_to_malicious_nodes = torch.load(out_file)
    for tw, nodes in tw_to_malicious_nodes.items():
        unique_nodes, counts = np.unique(nodes, return_counts=True)
        node_to_count = {node: count for node, count in zip(unique_nodes, counts)}
        log(f"TW {tw} -> {len(unique_nodes)} malicious nodes + {len(nodes)} malicious edges")

        node_to_count = {uuid_to_node_id[node_id]: count for node_id, count in node_to_count.items()}
        pprint(node_to_count, width=1)
        tw_to_malicious_nodes[tw] = node_to_count

    return tw_to_malicious_nodes

def viz_graph(
    edge_index,
    edge_scores,
    node_scores,
    node_to_correct_pred,
    malicious_nodes,
    node_to_path_and_type,
    anomaly_threshold,
    out_dir,
    tw,
    cfg,
    n_hop,
    fuse_nodes,
):
    # On OpTC, the degree is too high so we remove 90% of non-malicious nodes
    # for visualization.
    # if dataset == "OPTC":
    #     OPTC_NODE_TO_VISUALIZE = 201  # Set the node viz manually
    #     idx = edge_index[0, :] == OPTC_NODE_TO_VISUALIZE

    #     if 1 not in y[idx]:
    #         return

    #     edge_index = edge_index[:, idx]
    #     edge_scores = edge_scores[idx]
    #     y = y[idx]

    #     indices_0 = np.where(y == 0)[0]
    #     indices_1 = np.where(y == 1)[0]

    #     selected_indices_0 = np.random.choice(
    #         indices_0, size=int(len(indices_0) * 0.1), replace=False
    #     )
    #     final_indices = np.concatenate((selected_indices_0, indices_1))
    #     np.random.shuffle(final_indices)

    #     edge_index = edge_index[:, final_indices]
    #     edge_scores = edge_scores[final_indices]
    #     y = y[final_indices]
    
    if edge_index.shape[0] != 2:
        edge_index = np.array([edge_index[:, 0], edge_index[:, 1]])

    if fuse_nodes:
        idx = 0
        merged_nodes = defaultdict(lambda: defaultdict(int))
        merged_edges = defaultdict(lambda: defaultdict(list))
        old_node_to_merged_node = defaultdict(list)
        for i, (src, dst, score) in enumerate(zip(edge_index[0], edge_index[1], edge_scores)):
            edge_tuple = []
            for node in [src, dst]:
                path = node_to_path_and_type[node]['path']
                typ = node_to_path_and_type[node]['type']
                
                if (path, typ) not in merged_nodes:
                    merged_nodes[(path, typ)] = {"idx": idx, "label": 0, "predicted": 0}
                    idx += 1
                edge_tuple.append(merged_nodes[(path, typ)]["idx"])
                old_node_to_merged_node[node] = merged_nodes[(path, typ)]["idx"]
                
                # If only one malicious node is present in the merged node, it is malicious
                merged_nodes[(path, typ)]["label"] = max(merged_nodes[(path, typ)]["label"], int(node in malicious_nodes))
                # I fonly one good prediction of the merged nodes is correct, we set predicted=1. If node not predicted, we set to -1
                merged_nodes[(path, typ)]["predicted"] = max(merged_nodes[(path, typ)]["predicted"], int(node_to_correct_pred.get(node, -1)))
            
            merged_edges[tuple(edge_tuple)]["t"].append(i)
            merged_edges[tuple(edge_tuple)]["score"].append(score)

        new_edge_index = np.array(list(merged_edges.keys())).T
        merged_edge_scores = [np.max(d["score"]) for _, d in merged_edges.items()]
        edge_t = [f"{np.min(d['t'])}-{np.max(d['t'])}" for _, d in merged_edges.items()]

        # sorted_merged_nodes = dict(sorted(merged_nodes.items(), key=lambda item: item[1]))
        unique_nodes, unique_labels, unique_predicted, unique_paths, unique_types = [], [], [], [], []
        for (path, typ), d in merged_nodes.items():
            unique_nodes.append(d["idx"])
            unique_labels.append(d["label"])
            unique_predicted.append(d["predicted"])
            unique_paths.append(path)
            unique_types.append(typ)
            
        source_nodes = malicious_nodes
        new_source_nodes = {old_node_to_merged_node[n] for n in source_nodes}

    else:
        # Flatten edge_index and map node IDs to a contiguous range starting from 0
        unique_nodes, new_edge_index = np.unique(edge_index.flatten(), return_inverse=True)
        new_edge_index = new_edge_index.reshape(edge_index.shape)
        unique_paths = [node_to_path_and_type[n]["path"] for n in unique_nodes]
        unique_types = [node_to_path_and_type[n]["type"] for n in unique_nodes]
        unique_labels = [n in malicious_nodes for n in unique_nodes]
        unique_predicted = [node_to_correct_pred.get(n, -1) for n in unique_nodes]
        edge_t = list(range(len(edge_index[0])))
        
        source_nodes = malicious_nodes
        source_node_map = {old: new for new, old in enumerate(unique_nodes)}
        new_source_nodes = [source_node_map.get(node, -1) for node in source_nodes]

    G = ig.Graph(edges=[tuple(e) for e in new_edge_index.T], directed=True)

    # Node attributes
    G.vs["original_id"] = unique_nodes
    G.vs["path"] = unique_paths
    G.vs["type"] = unique_types
    G.vs["shape"] = ["rectangle" if typ == "file" else "circle" if typ == "subject" else "triangle" for typ in unique_types]
    
    G.vs["label"] = unique_labels
    G.vs["predicted"] = unique_predicted
    G.es["t"] = edge_t

    # Edge attributes
    G.es["anomaly_score"] = edge_scores

    # Find N-hop neighborhoods for the source nodes
    neighborhoods = set()
    for node in new_source_nodes:
        if node == -1:
            # Warning: one malicious node in {source_nodes} was not seen in the dataset ({new_source_nodes}).
            continue
        neighborhood = G.neighborhood(node, order=n_hop)
        neighborhoods.update(neighborhood)

    # Create a subgraph with only the n-hop neighborhoods
    subgraph = G.subgraph(neighborhoods)

    BENIGN = "#44BC"
    ATTACK = "#FF7E79"
    FAILURE = "red"
    SUCCESS = "green"

    visual_style = {}
    visual_style["bbox"] = (700, 700)
    visual_style["margin"] = 40
    visual_style["layout"] = subgraph.layout("kk")

    visual_style["vertex_size"] = 13
    visual_style["vertex_width"] = 13
    visual_style["vertex_label_dist"] = 1.3
    visual_style["vertex_label_size"] = 6
    visual_style["vertex_label_font"] = 1
    visual_style["vertex_color"] = [ATTACK if label else BENIGN for label in subgraph.vs["label"]]
    visual_style["vertex_label"] = subgraph.vs["path"]
    visual_style["vertex_frame_width"] = 2
    visual_style["vertex_frame_color"] = ["black" if predicted == -1 else SUCCESS if predicted else FAILURE for predicted in subgraph.vs["predicted"]]

    visual_style["edge_curved"] = 0.1
    visual_style["edge_width"] = 1 #[3 if label else 1 for label in y_hat]
    visual_style["edge_color"] = "gray" # ["red" if label else "gray" for label in subgraph.es["y"]]
    visual_style["edge_label"] = [f"s:{x:.2f}\nt:{t}" for x, t in zip(subgraph.es["anomaly_score"], subgraph.es["t"])]
    visual_style["edge_label_size"] = 6
    visual_style["edge_label_color"] = "#888888"
    visual_style["edge_arrow_size"] = 8
    visual_style["edge_arrow_width"] = 8

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the graph using igraph
    plot = ig.plot(subgraph, target=ax, **visual_style)

    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=BENIGN, label='Benign'),
        mpatches.Patch(color=ATTACK, label='Attack'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10, label='Subject', markeredgewidth=1),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=10, label='File', markeredgewidth=1),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=10, label='IP', markeredgewidth=1),
        mpatches.Patch(edgecolor=FAILURE, label='False Pos/Neg', facecolor='none'),
        mpatches.Patch(edgecolor=SUCCESS, label='True Pos/Neg', facecolor='none')
    ]

    # Add legend to the plot
    ax.legend(handles=legend_handles, loc='upper right', fontsize='medium')

    # Save the plot with legend
    out_file = f"{n_hop}-hop_attack_graph_tw_{tw}"
    svg = os.path.join(out_dir, f"{out_file}.png")
    plt.savefig(svg)
    plt.close(fig)

    print(f"Graph {svg} saved, with attack nodes:\t {','.join([str(n) for n in source_nodes])}.")
    return {out_file: wandb.Image(svg)}

def compute_kmeans_labels(results, topk_K):
    nodes_to_score = sorted([(node_id, d["score"]) for node_id, d in results.items()], key=lambda x: x[1])
    nodes_to_score = np.array(nodes_to_score, dtype=object)
    score_values = nodes_to_score[:, 1].astype(float)

    last_N_scores = score_values[-topk_K:]
    last_N_nodes = nodes_to_score[-topk_K:]

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(last_N_scores.reshape(-1, 1))

    centroids = kmeans.cluster_centers_.flatten()
    highest_cluster_index = np.argmax(centroids)

    # Extract scores from the highest value cluster
    highest_value_cluster_indices = np.where(kmeans.labels_ == highest_cluster_index)[0]
    highest_value_cluster = last_N_nodes[highest_value_cluster_indices]

    # Extract scores and nodes from the highest cluster
    cluster_scores = highest_value_cluster[:, 1].astype(float)
    anomaly_nodes = highest_value_cluster[:, 0]
    
    for idx in highest_value_cluster_indices:
        global_idx = len(score_values) - topk_K + idx
        node_id = nodes_to_score[global_idx, 0]
        results[node_id]["y_hat"] = 1
        
    return results
