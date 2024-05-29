from collections import defaultdict

from provnet_utils import *
from config import *
from .evaluation_utils import *
from .node_evaluation import get_node_thr


def get_tw_scores(tw_path, cfg):
    log(f"Loading data from {tw_path}...")
    
    edge_index, losses, item_to_scores = defaultdict(list), defaultdict(list), defaultdict(list)

    filelist = listdir_sorted(tw_path)
    for tw, file in enumerate(tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(tw_path, file)
        with open(file, 'r') as f:
            for line in f:
                l = line.strip()
                data = eval(l)
                srcnode = data['srcnode']
                dstnode = data['dstnode']
                loss = data['loss']
                
                edge_index[tw].append((srcnode, dstnode))
                losses[tw].append(loss)
                item_to_scores[tw].append(loss)

    return edge_index, losses, item_to_scores

def main(val_tw_path, tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    edge_index, losses, item_to_scores = get_tw_scores(tw_path, cfg)
    tw_labels = set(tw_to_malicious_nodes.keys())

    os.makedirs(cfg.detection.evaluation.node_evaluation._precision_recall_dir, exist_ok=True)
    pr_img_file = os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"{model_epoch_dir}.png")
    scores_img_file = os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"scores_{model_epoch_dir}.png")
    
    thr = get_node_thr(val_tw_path, cfg)
    log(f"Threshold: {thr:.3f}")
    
    y_truth, y_pred, pred_scores = [], [], []
    for tw, loss_list in item_to_scores.items():
        pred_score = None
        if cfg.detection.evaluation.tw_evaluation.use_mean_node_loss:
            pred_score = np.mean(loss_list)
        else:
            pred_score = np.max(loss_list)

        y_truth.append(tw in tw_labels)
        y_pred.append(int(pred_score > thr))
        pred_scores.append(pred_score)
        if tw in tw_labels:
            log(f"-> Malicious TW {tw}: loss={pred_score:.3f}" + (" ✅" if pred_score > thr else " ❌"))
            
    # Plots the PR curve and scores for mean node loss
    plot_precision_recall(pred_scores, y_truth, pr_img_file)
    plot_scores(pred_scores, y_truth, scores_img_file)
    return classifier_evaluation(y_truth, y_pred, pred_scores)
