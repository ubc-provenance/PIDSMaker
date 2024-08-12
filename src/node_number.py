import argparse

import torch
import wandb
from config import *
from provnet_utils import *

from preprocessing import (
    build_graphs,
)

from config import (
    get_yml_cfg,
    get_runtime_required_args,
)

def compute_node_number(split_files):
    all_nids = set()
    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)
    for graph_path in sorted_paths:
        graph = torch.load(graph_path)
        all_nids |= set(graph.nodes())
    return len(all_nids)

def main(cfg, **kwargs):
    modified_tasks = {subtask: restart for subtask, restart in cfg._subtasks_should_restart}
    should_restart = {subtask: restart for subtask, restart in cfg._subtasks_should_restart_with_deps}
    
    log("\n" + ("*" * 100))
    log("Tasks modified since last runs:")
    log("  =>  ".join([f"{subtask}({restart})" for subtask, restart in modified_tasks.items()]))

    log("\nTasks requiring re-execution:")
    log("  =>  ".join([f"{subtask}({restart})" for subtask, restart in should_restart.items()]))
    log(("*" * 100) + "\n")


    if should_restart["build_graphs"]:
        build_graphs.main(cfg)

    node_num_train = compute_node_number(split_files=cfg.dataset.train_files)
    node_num_val = compute_node_number(split_files=cfg.dataset.val_files)
    node_num_test = compute_node_number(split_files=cfg.dataset.test_files)
    node_num_unused = compute_node_number(split_files=cfg.dataset.unused_files)

    node_number = {
        'train node number': node_num_train,
        'val node number': node_num_val,
        'test node number': node_num_test,
        'unused node number': node_num_unused,
    }

    for k, v in node_number.items():
        log(f"{k}: {v}")

    wandb.log(node_number)

if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    
    exp_name = args.exp if args.exp != "" else \
        "|".join([f"{k.split('.')[-1]}={v}" for k, v in args.__dict__.items() if "." in k and v is not None])
    tags = args.tags.split(",") if args.tags != "" else [args.model]
    
    wandb.init(
        mode="online" if args.wandb else "disabled",
        # project="Orthrus_V1_bis",
        project="orthrus_evaluation",
        name=exp_name,
        tags=tags,
    )
    
    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg)
    
    wandb.finish()
