import os
import wandb
import numpy as np
from collections import defaultdict
from pprint import pprint

from .evaluation_methods import (
    node_evaluation,
    queue_evaluation,
    tw_evaluation,
    node_tw_evaluation,
)
from data_utils import *
from provnet_utils import log
from .evaluation_methods.evaluation_utils import *


def standard_evaluation(cfg, evaluation_fn):
    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "val")
    
    tw_to_malicious_nodes = compute_tw_labels(cfg)
    
    best_ap, best_stats = 0.0, {}
    for model_epoch_dir in listdir_sorted(test_losses_dir):
        log(f"\nEvaluation of model {model_epoch_dir}...")

        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        val_tw_path = os.path.join(val_losses_dir, model_epoch_dir)

        stats = evaluation_fn(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes=tw_to_malicious_nodes)
            
        stats["epoch"] = int(model_epoch_dir.split("_")[-1])
        stats["precision_recall_img"] = wandb.Image(os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"{model_epoch_dir}.png"))
        stats["scores_img"] = wandb.Image(os.path.join(cfg.detection.evaluation.node_evaluation._precision_recall_dir, f"scores_{model_epoch_dir}.png"))
        
        wandb.log(stats)
        
        if stats["ap"] > best_ap:
            best_ap = stats["ap"]
            best_stats = stats
        
    wandb.log(best_stats)


def main(cfg):
    method = cfg.detection.evaluation.used_method.strip()
    if method == "node_evaluation":
        standard_evaluation(cfg, evaluation_fn=node_evaluation.main)
    elif method == "tw_evaluation":
        standard_evaluation(cfg, evaluation_fn=tw_evaluation.main)
    elif method == "node_tw_evaluation":
        standard_evaluation(cfg, evaluation_fn=node_tw_evaluation.main)
    elif method == "queue_evaluation":
        queue_evaluation.main(cfg)
    else:
        raise ValueError(f"Invalid evaluation method {cfg.detection.evaluation.used_method}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
