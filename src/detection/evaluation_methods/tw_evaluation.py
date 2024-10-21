from collections import defaultdict

from provnet_utils import *
from config import *
from .evaluation_utils import *


def get_tw_predictions(val_tw_path, test_tw_path, cfg, tw_to_malicious_nodes):
    log(f"Loading data from {test_tw_path}...")
    
    thr = get_threshold(val_tw_path, cfg.detection.evaluation.tw_evaluation.threshold_method)
    log(f"Threshold: {thr:.3f}")
    
    tw_to_losses = defaultdict(list)

    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient='records')
        for line in df:
            srcnode = line['srcnode']
            dstnode = line['dstnode']
            loss = line['loss']
            
            tw_to_losses[tw].append(loss)
                
    tw_labels = set(tw_to_malicious_nodes.keys())
    results = defaultdict(dict)
    for tw, losses in tw_to_losses.items():
        pred_score = reduce_losses_to_score(losses, cfg.detection.evaluation.tw_evaluation.threshold_method)
            
        results[tw]["score"] = pred_score
        results[tw]["y_hat"] = int(pred_score > thr)
        results[tw]["y_true"] = int(tw in tw_labels)

    return results

def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    results = get_tw_predictions(val_tw_path, test_tw_path, cfg, tw_to_malicious_nodes)

    out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    pr_img_file = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
    simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
    dor_img_file = os.path.join(out_dir, f"dor_{model_epoch_dir}.png")
    
    y_truth, y_preds, pred_scores = [], [], []
    for tw, result in results.items():
        score, y_hat, y_true = result["score"], result["y_hat"], result["y_true"]
        y_truth.append(y_true)
        y_preds.append(y_hat)
        pred_scores.append(score)

        if y_true == 1:
            log(f"-> Malicious TW {tw}: loss={score:.3f} | is TP:" + (" ✅" if y_true == y_hat else " ❌"))
            
    # Plots the PR curve and scores for mean node loss
    log(f"Saving figures to {out_dir}...")
    plot_precision_recall(pred_scores, y_truth, pr_img_file)
    plot_dor_recall_curve(pred_scores, y_truth, dor_img_file)
    plot_simple_scores(pred_scores, y_truth, simple_scores_img_file)
    return classifier_evaluation(y_truth, y_preds, pred_scores)
