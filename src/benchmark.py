import argparse
import random

import torch
import wandb
import numpy as np
from provnet_utils import remove_underscore_keys, log

from preprocessing import (
    build_graphs,
)
from featurization import (
    embed_edges,
    embed_nodes,
)
from detection import (
    gnn_training,
    gnn_testing,
    evaluation,
)

from config import (
    get_yml_cfg,
    get_runtime_required_args,
)

from triage import (
    tracing,
)

import time

def main(cfg, **kwargs):
    modified_tasks = {subtask: restart for subtask, restart in cfg._subtasks_should_restart}
    should_restart = {subtask: restart for subtask, restart in cfg._subtasks_should_restart_with_deps}
    
    log("\n" + ("*" * 100))
    log("Tasks modified since last runs:")
    log("  =>  ".join([f"{subtask}({restart})" for subtask, restart in modified_tasks.items()]))

    log("\nTasks requiring re-execution:")
    log("  =>  ".join([f"{subtask}({restart})" for subtask, restart in should_restart.items()]))
    log(("*" * 100) + "\n")
    
    if cfg.detection.gnn_training.use_seed:
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    t0 = time.time()

    # Preprocessing
    if should_restart["build_graphs"]:
        build_graphs.main(cfg)

    t1 = time.time()
    
    # Featurization
    if should_restart["embed_nodes"]:
        embed_nodes.main(cfg)
    t2 = time.time()
    if should_restart["embed_edges"]:
        embed_edges.main(cfg)
    t3 = time.time()

    # Detection
    if should_restart["gnn_training"]:
        gnn_training.main(cfg, **kwargs)
        torch.cuda.empty_cache()
    t4 = time.time()
    if should_restart["gnn_testing"]:
        gnn_testing.main(cfg)
    t5 = time.time()
    if should_restart["evaluation"]:
        evaluation.main(cfg)
    t6 = time.time()

    # Triage
    if should_restart["tracing"] and cfg.triage.tracing.used_method is not None:
        if cfg.detection.evaluation.used_method.strip() in ['node_evaluation', 'node_tw_evaluation']:
            tracing.main(cfg)
    t7 = time.time()

    time_consumption = {
        "time_total": round(t7 - t0, 2),
        "time_build_graphs": round(t1 - t0, 2),
        "time_embed_nodes": round(t2 - t1, 2),
        "time_embed_edges": round(t3 - t2, 2),
        "time_gnn_training": round(t4 - t3, 2),
        "time_gnn_testing": round(t5 - t4, 2),
        "time_evaluation": round(t6 - t5, 2),
        "time_tracing": round(t7 - t6, 2),
    }

    log("==" * 30)
    log("Run finished. Time consumed in each step:")
    for k, v in time_consumption.items():
        log(f"{k}: {v} s")

    log("==" * 30)
    wandb.log(time_consumption)


if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    
    exp_name = args.exp if args.exp != "" else \
        args.__dict__["dataset"]
        # "|".join([f"{k.split('.')[-1]}={v}" for k, v in args.__dict__.items() if "." in k and v is not None])
    tags = args.tags.split(",") if args.tags != "" else [args.model]
    
    wandb.init(
        mode="online" if args.wandb else "disabled",
        # project="Orthrus_V1_bis",
        project="kairos_evaluation",
        name=exp_name,
        tags=tags,
    )
    
    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg)
    
    wandb.finish()
