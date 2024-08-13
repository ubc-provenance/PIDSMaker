from data_utils import *
from provnet_utils import *

from .evaluation_utils import compute_tw_labels_for_magic

from magic_utils import magic_eval
import os
import numpy as np
from .evaluation_utils import *
import torch
import wandb
from labelling import get_ground_truth

def classifier_node_evaluation(y_test, y_test_pred):
    labels_exist = sum(y_test) > 0
    if labels_exist:
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    else:
        tn, fp, fn, tp = 1, 1, 1, 1  # only to not break tests

    fpr = fp / (fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fscore = 2 * (precision * recall) / (precision + recall)

    log(f'total num: {len(y_test)}')
    log(f'tn: {tn}')
    log(f'fp: {fp}')
    log(f'fn: {fn}')
    log(f'tp: {tp}')
    log('')

    log(f"precision: {precision}")
    log(f"recall: {recall}")
    log(f"fpr: {fpr}")
    log(f"fscore: {fscore}")
    log(f"accuracy: {accuracy}")


    stats = {
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "fpr": round(fpr, 7),
        "fscore": round(fscore, 5),
        "accuracy": round(accuracy, 5),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    return stats

def main(cfg):
    log("Get ground truth")
    GP_nids, _, _ = get_ground_truth(cfg)
    GPs = [str(nid) for nid in GP_nids]

    tw_to_malicious_nodes = compute_tw_labels_for_magic(cfg)

    results, node_to_max_loss_tw = magic_eval.get_node_predictions(cfg, tw_to_malicious_nodes)

    model_epoch_dir = "magic_evaluation"

    save_dir = cfg.detection.evaluation._evaluation_results_dir
    os.makedirs(save_dir, exist_ok=True)

    log("Get all nodes")
    all_nids = set()
    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    split_files = cfg.dataset.test_files
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    for graph_path in sorted_paths:
        graph = torch.load(graph_path)
        all_nids |= set(graph.nodes())

    log("Get model positives")
    MPs = set()
    for tw, nid2data in results.items():
        for node_id, data in nid2data.items():
            if data['y_hat'] == 1:
                MPs.add(str(node_id))

    nodes, y_hat, y_truth = [], [], []

    for n in all_nids:
        nodes.append(n)
        y_hat.append(int(n in MPs))
        y_truth.append(int(n in GPs))

    log(f"Results of epoch {model_epoch_dir}")
    log("==" * 30)
    stats = classifier_node_evaluation(y_truth, y_hat)
    log("==" * 30)

    results_file = os.path.join(save_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(save_dir, f"stats_{model_epoch_dir}.pth")

    torch.save(results, results_file)
    torch.save(stats, stats_file)

    stats["epoch"] = cfg.featurization.embed_edges.magic.max_epoch

    wandb.log(stats)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)