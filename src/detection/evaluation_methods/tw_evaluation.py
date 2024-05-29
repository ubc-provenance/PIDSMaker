from collections import defaultdict

from provnet_utils import *
from config import *
from .evaluation_utils import *
from .node_evaluation import get_node_thr


def get_tw_predictions(val_tw_path, test_tw_path, cfg, tw_to_malicious_nodes):
    log(f"Loading data from {test_tw_path}...")
    
    thr = get_node_thr(val_tw_path, cfg)
    log(f"Threshold: {thr:.3f}")
    
    tw_to_losses = defaultdict(list)

    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        with open(file, 'r') as f:
            for line in f:
                l = line.strip()
                data = eval(l)
                srcnode = data['srcnode']
                dstnode = data['dstnode']
                loss = data['loss']
                
                tw_to_losses[tw].append(loss)
                
    tw_labels = set(tw_to_malicious_nodes.keys())
    results = defaultdict(dict)
    for tw, losses in tw_to_losses.items():
        pred_score = None
        if cfg.detection.evaluation.tw_evaluation.use_mean_node_loss:
            pred_score = np.mean(losses)
        else:
            pred_score = np.max(losses)
            
        results[tw]["score"] = pred_score
        results[tw]["y_hat"] = int(pred_score > thr)
        results[tw]["y_true"] = int(tw in tw_labels)

    return results

def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    results = get_tw_predictions(val_tw_path, test_tw_path, cfg, tw_to_malicious_nodes)

    os.makedirs(cfg.detection.evaluation.node_evaluation._precision_recall_dir, exist_ok=True)
    pr_img_file = os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"{model_epoch_dir}.png")
    scores_img_file = os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"scores_{model_epoch_dir}.png")
    fp_img_file = os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"false_positives_{model_epoch_dir}.png")
    
    y_truth, y_preds, pred_scores = [], [], []
    for tw, result in results.items():
        score, y_hat, y_true = result["score"], result["y_hat"], result["y_true"]
        y_truth.append(y_true)
        y_preds.append(y_hat)
        pred_scores.append(score)

        if y_true == 1:
            log(f"-> Malicious TW {tw}: loss={score:.3f} | is TP:" + (" ✅" if y_true == y_hat else " ❌"))
            
    # Plots the PR curve and scores for mean node loss
    plot_precision_recall(pred_scores, y_truth, pr_img_file)
    plot_scores(pred_scores, y_truth, scores_img_file)
    plot_false_positives(y_truth, y_preds, fp_img_file)
    return classifier_evaluation(y_truth, y_preds, pred_scores)
