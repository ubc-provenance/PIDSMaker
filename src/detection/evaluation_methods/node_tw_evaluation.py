import os
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from provnet_utils import *
from config import *
from .evaluation_utils import *


def get_node_predictions(val_tw_path, test_tw_path, cfg, tw_to_malicious_nodes):
    ground_truth_nids, ground_truth_paths = get_ground_truth_nids(cfg)
    log(f"Calculating threshold...")
    
    thr = get_threshold(val_tw_path, cfg.detection.evaluation.node_tw_evaluation.threshold_method)
    log(f"Threshold: {thr:.3f}")

    tw_to_node_to_losses = defaultdict(lambda: defaultdict(list))
    tw_to_edge_index = defaultdict(list)
    tw_to_edge_loss = defaultdict(list)
    node_to_max_loss_tw = {}
    node_to_max_loss = defaultdict(int)
    
    filelist = listdir_sorted(test_tw_path)
    for tw, file in enumerate(tqdm(sorted(filelist), desc="Compute labels")):
        file = os.path.join(test_tw_path, file)
        df = pd.read_csv(file).to_dict(orient='records')
        for line in df:
            srcnode = line['srcnode']
            dstnode = line['dstnode']
            loss = line['loss']
            
            tw_to_edge_index[tw].append((srcnode, dstnode))
            tw_to_edge_loss[tw].append(loss)
            
            # Scores
            tw_to_node_to_losses[tw][srcnode].append(loss) # TODO: now we only consider src nodes and we don't evaluate on dst nodes
            if cfg.detection.evaluation.node_tw_evaluation.use_dst_node_loss:
                tw_to_node_to_losses[tw][dstnode].append(loss)
                
            # If max-val thr is used, we want to keep track when the node with max loss happens
            if loss > node_to_max_loss[srcnode]:
                node_to_max_loss[srcnode] = loss
                node_to_max_loss_tw[srcnode] = tw
            if cfg.detection.evaluation.node_tw_evaluation.use_dst_node_loss:
                if loss > node_to_max_loss[dstnode]:
                    node_to_max_loss[dstnode] = loss
                    node_to_max_loss_tw[dstnode] = tw

    use_kmeans = cfg.detection.evaluation.node_tw_evaluation.use_kmeans
    results = {}
    for tw, node_to_losses in tw_to_node_to_losses.items():
        is_malicious_tw = False
        
        if tw not in results:
            results[tw] = {}
        for node_id, losses in node_to_losses.items():
            pred_score = reduce_losses_to_score(losses, cfg.detection.evaluation.node_tw_evaluation.threshold_method)

            if node_id not in results[tw]:
                results[tw][node_id] = {}

            results[tw][node_id]["score"] = pred_score
            results[tw][node_id]["y_true"] = int((tw in tw_to_malicious_nodes) and (str(node_id) in tw_to_malicious_nodes[tw]))
            
            if use_kmeans: # in this mode, we add the label after
                results[tw][node_id]["y_hat"] = 0
                if int(pred_score > thr):
                    is_malicious_tw = True
            else:
                results[tw][node_id]["y_hat"] = int(pred_score > thr)
                
        if is_malicious_tw:
            results[tw] = compute_kmeans_labels(results[tw], topk_K=cfg.detection.evaluation.node_tw_evaluation.kmeans_top_K)

    return results, tw_to_edge_index, tw_to_edge_loss, thr, node_to_max_loss_tw

def main(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes, **kwargs):
    results, tw_to_ei, tw_to_edge_loss, thr, node_to_max_loss_tw = get_node_predictions(val_tw_path, test_tw_path, cfg, tw_to_malicious_nodes)
    node_to_path = get_node_to_path_and_type(cfg)

    out_dir = cfg.detection.evaluation.node_tw_evaluation._precision_recall_dir
    os.makedirs(out_dir, exist_ok=True)
    pr_img_file = os.path.join(out_dir, f"{model_epoch_dir}.png")
    scores_img_file = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
    node_to_path_type = get_node_to_path_and_type(cfg)
    
    log("Analysis of malicious nodes:")
    nodes, y_truth, y_preds, pred_scores = [], [], [], []
    node_to_correct_pred = {}
    summary_graphs = {}
    
    for tw, nid_to_result in results.items():
        malicious_tws = set()
        malicious_nodes = set()
        
        # We create a new arrayfor each TW
        for arr in [nodes, y_truth, y_preds, pred_scores]:
            arr.append([])
        for nid, result in nid_to_result.items():
            nodes[tw].append(nid)
            score, y_hat, y_true = result["score"], result["y_hat"], result["y_true"]
            y_truth[tw].append(y_true)
            y_preds[tw].append(y_hat)
            pred_scores[tw].append(score)
            node_to_correct_pred[nid] = y_hat == y_true
            
            if y_true == 1:
                if tw not in malicious_tws:
                    log(f"TW {tw}")
                    malicious_tws.add(tw)
                log(f"-> Malicious node {nid:<7}: loss={score:.3f} | is TP:" + (" ✅ " if y_true == y_hat else " ❌ ") + (node_to_path[nid]['path']))
                malicious_nodes.add(nid)

        if cfg.detection.evaluation.viz_malicious_nodes and len(malicious_nodes) > 0:
            graph_path = viz_graph(
                edge_index=np.array(tw_to_ei[tw]),
                edge_scores=np.array(tw_to_edge_loss[tw]),
                node_scores=np.array(pred_scores[tw]),
                node_to_correct_pred=node_to_correct_pred,
                malicious_nodes=malicious_nodes,
                node_to_path_and_type=node_to_path_type,
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
    flat_nodes = [e for sublist in nodes for e in sublist]
    
    # Plots the PR curve and scores for mean node loss
    plot_precision_recall(flat_pred_scores, flat_y_truth, pr_img_file)
    
    max_val_loss_tw = [node_to_max_loss_tw[n] for n in flat_nodes]
    plot_scores_with_paths(flat_pred_scores, flat_y_truth, flat_nodes, max_val_loss_tw, tw_to_malicious_nodes, scores_img_file, cfg)
    stats = classifier_evaluation(flat_y_truth, flat_y_preds, flat_pred_scores)
    stats.update(**summary_graphs)

    results_file = os.path.join(out_dir, f"result_{model_epoch_dir}.pth")
    stats_file = os.path.join(out_dir, f"stats_{model_epoch_dir}.pth")

    torch.save(results, results_file)
    torch.save(stats, stats_file)
    
    return stats
