import argparse
import random

import torch
import wandb
import numpy as np
from provnet_utils import remove_underscore_keys, log
from config import set_task_to_done

from preprocessing import (
    build_graphs,
    transformation,
)
from featurization import (
    embed_edges,
    embed_nodes,
)
from detection import (
    gnn_training,
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

    task_to_module = {
        "build_graphs": {
            "module": build_graphs,
            "task_path": cfg.preprocessing.build_graphs._task_path,
        },
        "transformation": {
            "module": transformation,
            "task_path": cfg.preprocessing.transformation._task_path,
        },
        "embed_nodes": {
            "module": embed_nodes,
            "task_path": cfg.featurization.embed_nodes._task_path,
        },
        "embed_edges": {
            "module": embed_edges,
            "task_path": cfg.featurization.embed_edges._task_path,
        },
        "gnn_training": {
            "module": gnn_training,
            "task_path": cfg.detection.gnn_training._task_path,
        },
        "evaluation": {
            "module": evaluation,
            "task_path": cfg.detection.evaluation._task_path,
        },
        "tracing": {
            "module": tracing,
            "task_path": cfg.triage.tracing._task_path,
        },
    }
    
    def run_task(task: str):
        module = task_to_module[task]["module"]
        task_path = task_to_module[task]["task_path"]
        
        start = time.time()
        return_value = None
        
        if should_restart[task]:
            return_value = module.main(cfg)
            set_task_to_done(task_path)
        
        return {"time": time.time() - start, "return": return_value}
    
    # Standard behavior: we run the whole pipeline
    if cfg.experiments.experiment.used_method == "none":
        task_results = {task: run_task(task) for task in task_to_module}
        
        metrics = task_results["evaluation"]["return"]
        wandb.log(metrics)
        
        times = {f"time_{task}": round(results["time"], 2) for task, results in task_results.items()}
        wandb.log(times)
        
    log("==" * 30)
    log("Run finished.")
    log("==" * 30)
    


if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    
    exp_name = args.exp if args.exp != "" else \
        args.__dict__["dataset"]
        # "|".join([f"{k.split('.')[-1]}={v}" for k, v in args.__dict__.items() if "." in k and v is not None])
    tags = args.tags.split(",") if args.tags != "" else [args.model]
    
    PROJECT_PREFIX = "framework_"
    wandb.init(
        mode="online" if args.wandb else "disabled",
        project=PROJECT_PREFIX + "nodlink_tests",
        name=exp_name,
        tags=tags,
    )
    
    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg)
    
    wandb.finish()
