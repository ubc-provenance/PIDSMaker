"""Node-level evaluation where a node is evaluated once per time window.

A same node appearing in multiple time windows is considered as multiple unique
samples (one per window). This is a more realistic setting for near real-time
detection than `node_evaluation`, where node scores are aggregated over the whole
test set.
"""

import os
from collections import defaultdict

import numpy as np
import torch

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    classifier_evaluation,
    compute_kmeans_labels,
    get_threshold,
    load_losses_per_time_window,
    plot_precision_recall,
    plot_simple_scores,
    reduce_losses_to_score,
    viz_graph,
)
from pidsmaker.utils.dataset_utils import get_node_to_path_and_type
from pidsmaker.utils.utils import log, log_tqdm


def get_node_predictions(val_tw_path, test_tw_path, cfg, tw_to_malicious_nodes):
    node_tw_cfg = cfg.evaluation.node_tw_evaluation
    threshold_method = node_tw_cfg.threshold_method

    log("Calculating threshold...")
    thr = get_threshold(val_tw_path, threshold_method)
    log(f"Threshold: {thr:.3f}")

    tw_to_node_to_losses = defaultdict(lambda: defaultdict(list))
    tw_to_edge_index = defaultdict(list)
    tw_to_edge_loss = defaultdict(list)
    node_to_max_loss_tw = {}
    node_to_max_loss = defaultdict(lambda: float("-inf"))

    def update_max(node, loss, tw):
        if loss > node_to_max_loss[node]:
            node_to_max_loss[node] = loss
            node_to_max_loss_tw[node] = tw

    _, tw_to_df = load_losses_per_time_window(test_tw_path, cfg)
    for tw, df in log_tqdm(tw_to_df.items(), desc="Compute labels"):
        # Node-level objectives produce one loss per node
        if "node" in df.columns:
            for node, loss in zip(df["node"].astype(int), df["loss"]):
                tw_to_node_to_losses[tw][node].append(loss)
                update_max(node, loss, tw)
            continue

        # Edge-level objectives produce one loss per edge
        for srcnode, dstnode, loss in zip(
            df["srcnode"].astype(int), df["dstnode"].astype(int), df["loss"]
        ):
            tw_to_edge_index[tw].append((srcnode, dstnode))
            tw_to_edge_loss[tw].append(loss)

            tw_to_node_to_losses[tw][srcnode].append(loss)
            update_max(srcnode, loss, tw)
            if node_tw_cfg.use_dst_node_loss:
                tw_to_node_to_losses[tw][dstnode].append(loss)
                update_max(dstnode, loss, tw)

    use_kmeans = node_tw_cfg.use_kmeans
    results = {}
    for tw, node_to_losses in sorted(tw_to_node_to_losses.items()):
        is_malicious_tw = False
        malicious_nodes_in_tw = tw_to_malicious_nodes.get(tw, {})

        results[tw] = {}
        for node_id, losses in node_to_losses.items():
            pred_score = float(reduce_losses_to_score(losses, threshold_method))
            y_hat = int(pred_score > thr)

            results[tw][node_id] = {
                "score": pred_score,
                "y_true": int(str(node_id) in malicious_nodes_in_tw),
                # with kmeans, labels are set after clustering
                "y_hat": 0 if use_kmeans else y_hat,
            }
            is_malicious_tw = is_malicious_tw or bool(y_hat)

        # Kmeans is only applied on TWs with at least one node above the threshold
        if use_kmeans and is_malicious_tw:
            results[tw] = compute_kmeans_labels(results[tw], topk_K=node_tw_cfg.kmeans_top_K)

    return results, tw_to_edge_index, tw_to_edge_loss, thr, node_to_max_loss_tw


def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    results, tw_to_ei, tw_to_edge_loss, thr, node_to_max_loss_tw = get_node_predictions(
        val_tw_path, test_tw_path, cfg, tw_to_malicious_nodes
    )
    node_to_path = get_node_to_path_and_type(cfg)

    out_dir = cfg.evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    pr_img_file = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
    simple_scores_img_file = os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")

    log("Analysis of malicious nodes:")
    # One list per TW, in the same order as `tws`
    tws, nodes, y_truth, y_preds, pred_scores = [], [], [], [], []
    summary_graphs = {}

    for tw, nid_to_result in sorted(results.items()):
        tws.append(tw)
        tw_nodes, tw_truth, tw_preds, tw_scores = [], [], [], []
        node_to_correct_pred = {}
        malicious_nodes = set()

        for nid, result in nid_to_result.items():
            score, y_hat, y_true = result["score"], result["y_hat"], result["y_true"]
            tw_nodes.append(nid)
            tw_truth.append(y_true)
            tw_preds.append(y_hat)
            tw_scores.append(score)
            node_to_correct_pred[nid] = y_hat == y_true

            if y_true == 1:
                if len(malicious_nodes) == 0:
                    log(f"TW {tw}")
                log(
                    f"-> Malicious node {nid:<7}: loss={score:.3f} | is TP:"
                    + (" ✅ " if y_true == y_hat else " ❌ ")
                    + node_to_path.get(nid, {}).get("path", "?")
                )
                malicious_nodes.add(nid)

        nodes.append(tw_nodes)
        y_truth.append(tw_truth)
        y_preds.append(tw_preds)
        pred_scores.append(tw_scores)

        if (
            cfg.evaluation.viz_malicious_nodes
            and len(malicious_nodes) > 0
            and len(tw_to_ei[tw]) > 0
        ):
            graph_path = viz_graph(
                edge_index=np.array(tw_to_ei[tw]).T,
                edge_scores=np.array(tw_to_edge_loss[tw]),
                node_scores=np.array(tw_scores),
                node_to_correct_pred=node_to_correct_pred,
                malicious_nodes=malicious_nodes,
                node_to_path_and_type=node_to_path,
                anomaly_threshold=thr,
                out_dir=out_dir,
                tw=tw,
                cfg=cfg,
                n_hop=1,
                fuse_nodes=True,
            )
            summary_graphs.update(**graph_path)

    flat_pred_scores = [e for sublist in pred_scores for e in sublist]
    flat_y_truth = [e for sublist in y_truth for e in sublist]
    flat_y_preds = [e for sublist in y_preds for e in sublist]

    if cfg.evaluation.save_figures is not False:  # unset: saved
        log(f"Saving figures to {out_dir}...")
        plot_precision_recall(flat_pred_scores, flat_y_truth, pr_img_file)
        plot_simple_scores(flat_pred_scores, flat_y_truth, simple_scores_img_file)

    stats = classifier_evaluation(flat_y_truth, flat_y_preds, flat_pred_scores)
    stats["threshold"] = thr
    stats["num_malicious_tws_detected"] = sum(
        int(any(p and t for p, t in zip(preds, truths))) for preds, truths in zip(y_preds, y_truth)
    )
    stats["num_malicious_tws"] = sum(int(any(truths)) for truths in y_truth)

    results_file = os.path.join(out_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")
    scores_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.pkl")

    torch.save(results, results_file)
    torch.save(stats, stats_file)
    torch.save(
        {
            "pred_scores": flat_pred_scores,
            "y_preds": flat_y_preds,
            "y_truth": flat_y_truth,
            "nodes": [(tw, n) for tw, tw_nodes in zip(tws, nodes) for n in tw_nodes],
        },
        scores_file,
    )
    stats["scores_file"] = scores_file
    stats.update(**summary_graphs)

    return stats
