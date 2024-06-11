from collections import defaultdict

from sklearn.metrics import (
    auc,
    roc_curve,
)
import matplotlib.pyplot as plt
import re

from provnet_utils import *
from data_utils import *
from config import *


def get_threshold(val_tw_path, threshold_method: str):
    threshold_method = threshold_method.strip()
    if threshold_method == "max_val_loss":
        return calculate_threshold(val_tw_path)['max']
    elif threshold_method == "mean_val_loss":
        return calculate_threshold(val_tw_path)['mean']
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
        f = open(os.path.join(val_tw_dir, file))
        for line in f:
            l = line.strip()
            jdata = eval(l)

            loss_list.append(jdata['loss'])

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

    # Creating the plot
    plt.figure(figsize=(10, 3))  # Width, height in inches
    plt.scatter(scores_0, y_zeros, color='green', label='Label 0')
    plt.scatter(scores_1, y_ones, color='red', label='Label 1')

    # Adding labels and title
    plt.xlabel('Scores')
    plt.ylabel('Labels')
    plt.yticks([0, 1], ['0', '1'])  # Set y-ticks to show label categories
    plt.title('Scatter Plot of Scores by Label')
    plt.legend()
    plt.savefig(out_file)

def plot_scores_with_paths(scores, y_truth, nodes, max_val_loss_tw, tw_to_malicious_nodes, out_file, cfg):
    node_to_path = get_node_to_path_and_type(cfg)
    paths = [node_to_path[n]["path"] for n in nodes]
    
    scores_0 = [score for score, label in zip(scores, y_truth) if label == 0]
    scores_1 = [score for score, label in zip(scores, y_truth) if label == 1]
    paths_0 = [path for path, label in zip(paths, y_truth) if label == 0]
    paths_1 = [path for path, label in zip(paths, y_truth) if label == 1]

    # Positions on the y-axis for the scatter plot (can be zero or any other constant)
    y_zeros = [0] * len(scores_0)  # All zeros at y=0
    y_ones = [1] * len(scores_1)  # All ones at y=1, you can also keep them at y=0 if you prefer

    # Creating the plot
    plt.figure(figsize=(12, 6))  # Increase the width to make space for text
    plt.scatter(scores_0, y_zeros, color='green', label='Label 0')
    plt.scatter(scores_1, y_ones, color='red', label='Label 1')

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

    # Select the top N scores
    top = combined_scores_sorted[:10]

    # Separate the top scores by their labels
    top_0 = [item for item in top if item[2] == 0]
    top_1 = [item for item in top if item[2] == 1]

    # Annotate the top scores for label 0
    for i, (score, path, _, max_tw_idx) in enumerate(top_0):
        y_position = 0 - (i * 0.1)  # Adjust y-position for each label to avoid overlap
        plt.text(max(scores) + 1, y_position, f"{path} ({score:.2f}): TW {max_tw_idx}", fontsize=8, va='center', ha='left', color='green')

    # Annotate the top scores for label 1
    for i, (score, path, _, max_tw_idx) in enumerate(top_1):
        y_position = 1 - (i * 0.1)  # Adjust y-position for each label to avoid overlap and add space between groups
        plt.text(max(scores) + 1, y_position, f"{path} ({score:.2f}): TW {max_tw_idx}", fontsize=8, va='center', ha='left', color='red')
        
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

def get_ground_truth_nids(cfg):
    ground_truth_nids, ground_truth_paths = [], {}
    with open(os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_relative_path_new), 'r') as f:
        for line in f:
            node_uuid, node_labels, node_id = line.replace(" ", "").strip().split(",")
            if node_id != 'node_id':
                ground_truth_nids.append(int(node_id))
                ground_truth_paths[int(node_id)] = node_labels
    return set(ground_truth_nids), ground_truth_paths

def get_uuid_to_node_id(cfg):
    uuid_to_node_id = {}
    with open(os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_relative_path_new), 'r') as f:
        for line in f:
            node_uuid, node_labels, node_id = line.replace(" ", "").strip().split(",")
            if node_id != 'node_id':
                uuid_to_node_id[node_uuid] = node_id
    return uuid_to_node_id

def compute_tw_labels(cfg):
    """
    Gets the malcious node IDs present in each time window.
    """
    out_path = cfg.preprocessing.build_graphs._tw_labels
    out_file = os.path.join(out_path, "tw_to_malicious_nodes.pkl")
    
    if not os.path.exists(out_file):
        log(f"Computing time-window labels...")
        os.makedirs(out_path, exist_ok=True)
        event_labels_path = os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_events_relative_path)
        
        t_to_node = {}
        with open(event_labels_path, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                uuid, edge_type, src_id, dst_id, t = line.strip().split(", ")
                t_to_node[int(t)] = src_id
                
        test_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="test")
        
        num_found_event_labels = 0
        tw_to_malicious_nodes = defaultdict(list)
        for i, tw in enumerate(test_data):
            start = tw.t.min().item()
            end = tw.t.max().item()
            
            for t, node_id in t_to_node.items():
                if start < t < end:
                    tw_to_malicious_nodes[i].append(node_id)
                    num_found_event_labels += 1
                    
        log(f"Found {num_found_event_labels}/{len(t_to_node)} edge labels.")
        torch.save(tw_to_malicious_nodes, out_file)
        
    # Used to retrieve node ID from node raw UUID
    node_labels_path = os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_events_relative_path)
    uuid_to_node_id = get_uuid_to_node_id(cfg)
    
    # Create a mapping TW number => malicious node IDs
    tw_to_malicious_nodes = torch.load(out_file)
    for tw, nodes in tw_to_malicious_nodes.items():
        unique_nodes, counts = np.unique(nodes, return_counts=True)
        node_to_count = {node: count for node, count in zip(unique_nodes, counts)}
        log(f"TW {tw} -> {len(unique_nodes)} malicious nodes + {len(nodes)} malicious edges")
        
        node_to_count = {uuid_to_node_id[node_id]: count for node_id, count in node_to_count.items()}
        pprint(node_to_count, width=1)
        
    return tw_to_malicious_nodes
