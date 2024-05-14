import argparse

import wandb
from provnet_utils import remove_underscore_keys

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


def main(cfg):
    modified_tasks = {subtask: restart for subtask, restart in cfg._subtasks_should_restart}
    should_restart = {subtask: restart for subtask, restart in cfg._subtasks_should_restart_with_deps}
    
    print("\n" + ("*" * 100))
    print("Tasks modified since last runs:")
    print("  =>  ".join([f"{subtask}({restart})" for subtask, restart in modified_tasks.items()]))

    print("\nTasks requiring re-execution:")
    print("  =>  ".join([f"{subtask}({restart})" for subtask, restart in should_restart.items()]))
    print(("*" * 100) + "\n")


    # Preprocessing
    if should_restart["build_graphs"]:
        build_graphs.main(cfg)
        print("Finished building graphs")
    
    # Featurization
    if should_restart["embed_nodes"]:
        embed_nodes.main(cfg)
        print("Finished embedding nodes")
    if should_restart["embed_edges"]:
        embed_edges.main(cfg)
        print("Finished embedding edges")

    # Detection
    if should_restart["gnn_training"]:
        gnn_training.main(cfg)
        print("Finished gnn_training")
    if should_restart["gnn_testing"]:
        gnn_testing.main(cfg)
        print("Finished gnn_testing")
    # if should_restart["evaluation"]:
    #     evaluation.main(cfg)
    #     print("Finished evaluation")


if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    
    wandb.init(
        mode="online" if args.wandb_exp != "" else "disabled",
        project="framework",
        name=args.wandb_exp,
        tags=args.wandb_tag.split(",") if args.wandb_tag != "" else None,
    )
    
    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg)
    
    wandb.finish()
