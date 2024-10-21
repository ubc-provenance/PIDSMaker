import argparse
import copy
import random
from collections import defaultdict

import torch
import wandb
import numpy as np
from provnet_utils import remove_underscore_keys, log
from config import set_task_to_done, get_updated_should_restart
from experiments import update_cfg_for_uncertainty_exp, fuse_hyperparameter_metrics

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
    
    def run_task(task: str, cfg):
        module = task_to_module[task]["module"]
        task_path = task_to_module[task]["task_path"]
        
        start = time.time()
        return_value = None
        
        should_restart = get_updated_should_restart(cfg)
        if should_restart[task]:
            return_value = module.main(cfg)
            set_task_to_done(task_path)
        
        return {"time": time.time() - start, "return": return_value}
    
    def run_pipeline(cfg):
        task_results = {task: run_task(task, cfg) for task in task_to_module}
        
        metrics = task_results["evaluation"]["return"]
        times = {f"time_{task}": round(results["time"], 2) for task, results in task_results.items()}
        return metrics, times
    
    # Standard behavior: we run the whole pipeline
    if cfg.experiments.experiment.used_method == "standard":
        log("Running pipeline in 'Standard' mode.")
        metrics, times = run_pipeline(cfg)
        wandb.log(metrics)
        wandb.log(times)
        
    elif cfg.experiments.experiment.used_method == "uncertainty":
        log("Running pipeline in 'Uncertainty' mode.")
        method_to_metrics = defaultdict(list)
        original_cfg = copy.deepcopy(cfg)
        
        for method in ["mc_dropout", "hyperparameter", "deep_ensemble", "bagged_ensemble"]:
            iterations = getattr(cfg.experiments.experiment.uncertainty, method).iterations
            log(f"[@method {method}] - Started")
            
            if method == "hyperparameter":
                hyperparameters = cfg.experiments.experiment.uncertainty.hyperparameter.hyperparameters
                hyperparameters = map(lambda x: x.strip(), hyperparameters.split(","))
                assert iterations % 2 != 0, f"The number of iterations for hyperparameters should be odd, found {iterations}"
                
                for hyper in hyperparameters:
                    log(f"[@hyperparameter {hyper}] - Started")
                    hyper_to_metrics = defaultdict(list)
                    for i in range(iterations):
                        cfg = update_cfg_for_uncertainty_exp(method, i, iterations, original_cfg, hyperparameter=hyper)
                        metrics, times = run_pipeline(cfg)
                        hyper_to_metrics[hyper].append(metrics)
                        
                metrics = fuse_hyperparameter_metrics(hyper_to_metrics)
                method_to_metrics[method].append(metrics)
            
            else:
                for i in range(iterations):
                    cfg = update_cfg_for_uncertainty_exp(method, i, iterations, original_cfg, hyperparameter=None)
                    metrics, times = run_pipeline(cfg)
                    method_to_metrics[method].append(metrics)
                    
                    # We force restart in some methods so we avoid forced restart for other methods
                    cfg._force_restart = ""
                    cfg._is_running_mc_dropout = False
            
        
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
