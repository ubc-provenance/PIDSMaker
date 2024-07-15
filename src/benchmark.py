import argparse

import torch
import wandb
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


def main(cfg, **kwargs):
    modified_tasks = {subtask: restart for subtask, restart in cfg._subtasks_should_restart}
    should_restart = {subtask: restart for subtask, restart in cfg._subtasks_should_restart_with_deps}
    
    log("\n" + ("*" * 100))
    log("Tasks modified since last runs:")
    log("  =>  ".join([f"{subtask}({restart})" for subtask, restart in modified_tasks.items()]))

    log("\nTasks requiring re-execution:")
    log("  =>  ".join([f"{subtask}({restart})" for subtask, restart in should_restart.items()]))
    log(("*" * 100) + "\n")


    # Preprocessing
    if should_restart["build_graphs"]:
        build_graphs.main(cfg)
    
    # Featurization
    if should_restart["embed_nodes"]:
        embed_nodes.main(cfg)
    if should_restart["embed_edges"]:
        embed_edges.main(cfg)

    # Detection
    if should_restart["gnn_training"]:
        gnn_training.main(cfg, **kwargs)
        torch.cuda.empty_cache()
    if should_restart["gnn_testing"]:
        gnn_testing.main(cfg)
    if should_restart["evaluation"]:
        evaluation.main(cfg)


if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    
    exp_name = args.exp if args.exp != "" else \
        "|".join([f"{k.split('.')[-1]}={v}" for k, v in args.__dict__.items() if "." in k and v is not None])
    tags = args.tags.split(",") if args.tags != "" else [args.model]
    
    wandb.init(
        mode="online" if args.wandb else "disabled",
        # project="jbx_tests_featurization_theia_e5",
        project="Orthrus_V1",
        name=exp_name,
        tags=tags,
    )
    
    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg)
    
    wandb.finish()
