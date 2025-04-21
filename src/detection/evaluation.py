import os
import wandb

from .evaluation_methods import (
    node_evaluation,
    queue_evaluation,
    tw_evaluation,
    node_tw_evaluation,
    edge_evaluation,
)
from data_utils import *
from provnet_utils import log
from .evaluation_methods.evaluation_utils import *


def standard_evaluation(cfg, evaluation_fn):
    test_losses_dir = os.path.join(cfg.detection.gnn_training._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.detection.gnn_training._edge_losses_dir, "val")
    
    tw_to_malicious_nodes = compute_tw_labels(cfg)
    
    best_metrics = {
        "adp_score": float("-inf"),
        "discrimination": float("-inf"),
        "best_stats": None,
    }
    
    sorted_files = listdir_sorted(test_losses_dir) if os.path.exists(test_losses_dir) else ["epoch_0"]
    out_dir = cfg.detection.evaluation._precision_recall_dir
    
    save_files_to_wandb = cfg._experiment != "uncertainty"
        
    for model_epoch_dir in sorted_files:
        log(f"[@{model_epoch_dir}] - Test Evaluation", pre_return_line=True)

        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        val_tw_path = os.path.join(val_losses_dir, model_epoch_dir)

        stats = evaluation_fn(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes=tw_to_malicious_nodes)
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
            
            seen_scores = os.path.join(out_dir, f"seen_score_{model_epoch_dir}.png")
            if os.path.exists(seen_scores):
                stats['seen_scores_img'] = wandb.Image(seen_scores)
                
            discrim = os.path.join(out_dir, f"discrim_curve_{model_epoch_dir}.png")
            if os.path.exists(discrim):
                stats["discrim_img"] = wandb.Image(discrim)
        
        wandb.log(stats)
        
        best_metrics = best_metric_pick_best_epoch(stats, best_metrics, cfg)
        
    if save_files_to_wandb:
        # We only store the scores for the best run
        # wandb.save(best_metrics["stats"]["scores_file"], out_dir)
        wandb.save(best_metrics["stats"]["neat_scores_img_file"], out_dir)
        
    
    return best_metrics["stats"]

def best_metric_pick_best_epoch(stats, best_metrics, cfg):
    best_model_selection = cfg.detection.evaluation.best_model_selection
    
    if best_model_selection == "best_adp":
        condition =  (stats["adp_score"] > best_metrics["adp_score"]) or \
            (stats["adp_score"] == best_metrics["adp_score"] and stats["discrimination"] > best_metrics["discrimination"])
    
    elif best_model_selection == "best_discrimination":
        condition =  (stats["discrimination"] > best_metrics["discrimination"])
    
    else:
        raise ValueError(f"Invalid best model selection {best_model_selection}")
    
    if condition:
        best_metrics["adp_score"] = stats["adp_score"]
        best_metrics["discrimination"] = stats["discrimination"]
        best_metrics["stats"] = stats
    return best_metrics
    

def main(cfg):
    method = cfg.detection.evaluation.used_method.strip()
    if method == "node_evaluation":
        return standard_evaluation(cfg, evaluation_fn=node_evaluation.main)
    elif method == "tw_evaluation":
        return standard_evaluation(cfg, evaluation_fn=tw_evaluation.main)
    elif method == "node_tw_evaluation":
        return standard_evaluation(cfg, evaluation_fn=node_tw_evaluation.main)
    elif method == "queue_evaluation":
        return queue_evaluation.main(cfg)
    elif method == "edge_evaluation":
        return standard_evaluation(cfg, evaluation_fn=edge_evaluation.main)
    else:
        raise ValueError(f"Invalid evaluation method {cfg.detection.evaluation.used_method}")
