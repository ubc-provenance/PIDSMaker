import json
import os

import wandb

from pidsmaker.detection.evaluation_methods import (
    edge_evaluation,
    host_two_stage_evaluation,
    node_evaluation,
    node_tw_evaluation,
    queue_evaluation,
    tw_evaluation,
)
from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    compute_tw_labels,
    listdir_sorted,
    load_losses_per_time_window,
)
from pidsmaker.utils.utils import log


def standard_evaluation(cfg, evaluation_fn, group_by_time_window=False):
    test_losses_dir = os.path.join(cfg.training._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.training._edge_losses_dir, "val")

    if not os.path.exists(test_losses_dir) or cfg._from_weights:
        sorted_files = ["model_epoch_0"]
    else:
        sorted_files = listdir_sorted(test_losses_dir)

    # Pass the first epoch's CSV directory so compute_tw_labels enumerates the
    # same files as node_evaluation, avoiding TW index mismatch when
    # intra-graph batching splits graphs into multiple sub-batches.
    first_epoch_losses_dir = os.path.join(test_losses_dir, sorted_files[0])
    if group_by_time_window:
        # TW-level evaluations re-group the loss files into the original time windows
        tw_names, _ = load_losses_per_time_window(first_epoch_losses_dir, cfg)
        tw_to_malicious_nodes = compute_tw_labels(cfg, tw_names=tw_names)
    else:
        tw_to_malicious_nodes = compute_tw_labels(cfg, losses_dir=first_epoch_losses_dir)

    best_metrics = {
        "val_score": float("-inf"),
        "adp_score": float("-inf"),
        "discrimination": float("-inf"),
        "ap@10": float("-inf"),
        "ap": float("-inf"),
        "stats": None,
    }

    out_dir = cfg.evaluation._precision_recall_dir

    save_files_to_wandb = cfg._experiment != "uncertainty"

    for model_epoch_dir in sorted_files:
        log(f"[@{model_epoch_dir}] - Test Evaluation", pre_return_line=True)

        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        val_tw_path = os.path.join(val_losses_dir, model_epoch_dir)

        stats = evaluation_fn(
            val_tw_path,
            test_tw_path,
            model_epoch_dir,
            cfg,
            tw_to_malicious_nodes=tw_to_malicious_nodes,
        )
        val_score_file = os.path.join(
            cfg.training._edge_losses_dir, "val_scores", f"{model_epoch_dir}.json"
        )
        if os.path.exists(val_score_file):
            with open(val_score_file) as f:
                stats["val_score"] = json.load(f)["val_score"]

        log(f"[@{model_epoch_dir}] - Stats")
        for k, v in stats.items():
            log(f"{k}: {v}")

        stats["epoch"] = int(model_epoch_dir.split("_")[-1])

        if save_files_to_wandb:
            # stats["simple_scores_img"] = wandb.Image(os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png"))

            scores = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
            if os.path.exists(scores):
                stats["scores_img"] = wandb.Image(scores)

            # pr = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
            # if os.path.exists(pr):
            #     stats["precision_recall_img"] = wandb.Image(pr)

            adp = os.path.join(out_dir, f"adp_curve_{model_epoch_dir}.png")
            if os.path.exists(adp):
                stats["adp_img"] = wandb.Image(adp)

            # seen_scores = os.path.join(out_dir, f"seen_score_{model_epoch_dir}.png")
            # if os.path.exists(seen_scores):
            #     stats["seen_scores_img"] = wandb.Image(seen_scores)

            discrim = os.path.join(out_dir, f"discrim_curve_{model_epoch_dir}.png")
            if os.path.exists(discrim):
                stats["discrim_img"] = wandb.Image(discrim)
                
            distrib = os.path.join(out_dir, f"distrib_{model_epoch_dir}.png")
            if os.path.exists(distrib):
                stats["distrib_img"] = wandb.Image(distrib)

        wandb.log(stats)

        best_metrics = best_metric_pick_best_epoch(stats, best_metrics, cfg)

    if save_files_to_wandb and "neat_scores_img_file" in best_metrics["stats"]:
        # We only store the scores for the best run
        # wandb.save(best_metrics["stats"]["scores_file"], out_dir)
        wandb.save(best_metrics["stats"]["neat_scores_img_file"], out_dir)

    log(f"Best scores file: {best_metrics['stats'].get('scores_file')}")
    log(f"Best model path: {best_metrics['stats'].get('model_path')}")
    
    return best_metrics["stats"]


def best_metric_pick_best_epoch(stats, best_metrics, cfg):
    best_model_selection = cfg.evaluation.best_model_selection

    # Last evaluated epoch (epochs are evaluated in increasing order)
    if best_model_selection == "last_epoch":
        if best_metrics["stats"] is None or stats["epoch"] >= best_metrics["stats"]["epoch"]:
            best_metrics["stats"] = stats
        return best_metrics

    # Selection on the MCC of the predictions on the test set (as the code of LARES)
    if best_model_selection == "best_mcc":
        mcc = stats.get("mcc", float("-inf"))
        mcc = float("-inf") if mcc != mcc else mcc
        if best_metrics["stats"] is None or mcc > best_metrics.get("mcc", float("-inf")):
            best_metrics["mcc"] = mcc
            best_metrics["stats"] = stats
        return best_metrics

    # Selection on validation data only (e.g. link prediction AUC + AP of the validation edges)
    if best_model_selection == "best_val_score":
        if "val_score" not in stats:
            raise ValueError(
                "best_model_selection=best_val_score requires an objective computing a validation "
                "score (e.g. predict_edge_contrastive)."
            )
        if best_metrics["stats"] is None or stats["val_score"] > best_metrics["val_score"]:
            best_metrics["val_score"] = stats["val_score"]
            best_metrics["stats"] = stats
        return best_metrics

    # ADP and discrimination are only computed by node/edge-level evaluations.
    # Other evaluations (tw, node_tw) fall back to selecting the epoch with best AP.
    if "adp_score" not in stats or "discrimination" not in stats:
        stats_ap = stats.get("ap", float("nan"))
        stats_ap = float("-inf") if stats_ap != stats_ap else stats_ap  # nan => -inf
        if best_metrics["stats"] is None or stats_ap > best_metrics["ap"]:
            best_metrics["ap"] = stats_ap
            best_metrics["stats"] = stats
        return best_metrics

    if best_model_selection == "best_adp":
        condition = (stats["adp_score"] > best_metrics["adp_score"]) or (
            stats["adp_score"] == best_metrics["adp_score"]
            and stats["discrimination"] > best_metrics["discrimination"]
        )

    elif best_model_selection == "best_discrimination":
        condition = stats["discrimination"] > best_metrics["discrimination"]

    elif best_model_selection == "best_ap@10":
        condition = (stats["ap@10"] > best_metrics["ap@10"]) or (
            stats["ap@10"] == best_metrics["ap@10"]
            and stats["discrimination"] > best_metrics["discrimination"]
        )

    else:
        raise ValueError(f"Invalid best model selection {best_model_selection}")

    if condition:
        best_metrics["adp_score"] = stats["adp_score"]
        best_metrics["discrimination"] = stats["discrimination"]
        best_metrics["ap@10"] = stats.get("ap@10", float("-inf"))
        best_metrics["stats"] = stats
    return best_metrics


def main(cfg):
    method = cfg.evaluation.used_method.strip()
    if method == "node_evaluation":
        return standard_evaluation(cfg, evaluation_fn=node_evaluation.main)
    elif method == "tw_evaluation":
        return standard_evaluation(
            cfg, evaluation_fn=tw_evaluation.main, group_by_time_window=True
        )
    elif method == "node_tw_evaluation":
        return standard_evaluation(
            cfg, evaluation_fn=node_tw_evaluation.main, group_by_time_window=True
        )
    elif method == "queue_evaluation":
        return queue_evaluation.main(cfg)
    elif method == "edge_evaluation":
        return standard_evaluation(cfg, evaluation_fn=edge_evaluation.main)
    elif method == "host_two_stage_evaluation":
        return standard_evaluation(cfg, evaluation_fn=host_two_stage_evaluation.main)
    else:
        raise ValueError(f"Invalid evaluation method {cfg.evaluation.used_method}")
