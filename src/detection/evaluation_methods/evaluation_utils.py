from sklearn.metrics import (
    auc,
    roc_curve,
)
import matplotlib.pyplot as plt
import re

from provnet_utils import *
from config import *


def calculate_max_val_loss_threshold(graph_dir):
    filelist = listdir_sorted(graph_dir)

    loss_list = []
    for file in sorted(filelist):
        f = open(os.path.join(graph_dir, file))
        for line in f:
            l = line.strip()
            jdata = eval(l)

            loss_list.append(jdata['loss'])

    thr = {
        'max': max(loss_list),
        'avg': mean(loss_list),
        'percentile_90': percentile_90(loss_list)
           }
    log(f"Thr = {thr}, Avg = {mean(loss_list)}, STD = {std(loss_list)}, MAX = {max(loss_list)}, 90 Percentile = {percentile_90(loss_list)}")

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

def plot_scores(scores, y_truth, out_file):
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

def get_ground_truth_nids(cfg):
    ground_truth_nids = []
    with open(os.path.join(cfg._ground_truth_dir, cfg.dataset.ground_truth_relative_path_new), 'r') as f:
        for line in f:
            node_uuid, node_labels, node_id = line.replace(" ", "").strip().split(",")
            if node_id != 'node_id':
                ground_truth_nids.append(int(node_id))
    return ground_truth_nids

def compute_tw_labels(cfg):
    out_path = cfg.preprocessing.build_graphs._tw_labels
    out_file = os.path.join(out_path, "tw_to_malicious_nodes.pkl")
    
    if not os.path.exists(out_file):
        log(f"Computing time-window labels...")
        os.makedirs(out_path)
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
    
    tw_to_malicious_nodes = torch.load(out_file)
    for tw, nodes in tw_to_malicious_nodes.items():
        unique_nodes, counts = np.unique(nodes, return_counts=True)
        node_to_count = {node: count for node, count in zip(unique_nodes, counts)}
        log(f"TW {tw} -> {len(unique_nodes)} malicious nodes + {len(nodes)} malicious edges")
        pprint(node_to_count)
        
    return tw_to_malicious_nodes
