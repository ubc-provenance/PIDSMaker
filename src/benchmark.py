import argparse
import copy
import random
from collections import defaultdict

import torch
import wandb
import numpy as np
from provnet_utils import remove_underscore_keys, log
from config import set_task_to_done, get_updated_should_restart
from experiments.uncertainty import *
from experiments.tuning import get_tuning_sweep_cfg, fuse_cfg_with_sweep_cfg

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

def get_task_to_module(cfg):
    return {
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

def main(cfg, sweep_cfg=None, **kwargs):
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

    def run_task(task: str, cfg):
        start = time.time()
        return_value = None
        
        # This updates all task paths
        should_restart = get_updated_should_restart(cfg)
        
        task_to_module = get_task_to_module(cfg)
        module = task_to_module[task]["module"]
        task_path = task_to_module[task]["task_path"]
        
        if should_restart[task]:
            return_value = module.main(cfg)
            set_task_to_done(task_path)
        
        return {"time": time.time() - start, "return": return_value}
    
    def run_pipeline(cfg):
        tasks = get_task_to_module(cfg).keys()
        task_results = {task: run_task(task, cfg) for task in tasks}
        
        metrics = task_results["evaluation"]["return"] or {}
        metrics = {
            **metrics,
            "val_ap": task_results["gnn_training"]["return"],
        }
        
        times = {f"time_{task}": round(results["time"], 2) for task, results in task_results.items()}
        return metrics, times
    
    # Fine-tuning mode
    if cfg._tuning_mode != "none":
        log("Running pipeline in 'Tuning' mode.")
        sweep_config = get_tuning_sweep_cfg(cfg)
        project = f"framework_tuning_{cfg._model}"
        sweep_id = wandb.sweep(sweep_config, project=project)
        
        def run_pipeline_from_sweep(cfg):
            with wandb.init():
                sweep_cfg = wandb.config
                cfg = fuse_cfg_with_sweep_cfg(cfg, sweep_cfg)
                metrics, times = run_pipeline(cfg)
                wandb.log(metrics)
                wandb.log(times)
        
        count = sweep_config["count"] if "count" in sweep_config else None
        wandb.agent(sweep_id, lambda: run_pipeline_from_sweep(cfg), count=count)
    
    else:
    
        # Standard behavior: we run the whole pipeline
        if cfg.experiment.used_method == "no_experiment":
            log("Running pipeline in 'Standard' mode.")
            metrics, times = run_pipeline(cfg)
            wandb.log(metrics)
            wandb.log(times)
            
        elif cfg.experiment.used_method == "uncertainty":
            log("Running pipeline in 'Uncertainty' mode.")
            method_to_metrics = defaultdict(list)
            original_cfg = copy.deepcopy(cfg)
            
            uncertainty_methods = get_uncertainty_methods_to_run(cfg)
            for method in uncertainty_methods:
                iterations = getattr(cfg.experiment.uncertainty, method).iterations
                log(f"[@method {method}] - Started", pre_return_line=True)
                
                if method == "hyperparameter":
                    hyperparameters = cfg.experiment.uncertainty.hyperparameter.hyperparameters
                    hyperparameters = map(lambda x: x.strip(), hyperparameters.split(","))
                    assert iterations % 2 != 0, f"The number of iterations for hyperparameters should be odd, found {iterations}"
                    
                    hyper_to_metrics = defaultdict(list)
                    for hyper in hyperparameters:
                        log(f"[@hyperparameter {hyper}] - Started", pre_return_line=True)
                        
                        for i in range(iterations):
                            log(f"[@iteration {i}]", pre_return_line=True)
                            cfg = update_cfg_for_uncertainty_exp(method, i, iterations, copy.deepcopy(original_cfg), hyperparameter=hyper)
                            metrics, times = run_pipeline(cfg)
                            hyper_to_metrics[hyper].append({**metrics, **times})
                            
                    metrics = fuse_hyperparameter_metrics(hyper_to_metrics)
                    method_to_metrics[method] = metrics
                
                else:
                    for i in range(iterations):
                        log(f"[@iteration {i}]", pre_return_line=True)
                        cfg = update_cfg_for_uncertainty_exp(method, i, iterations, copy.deepcopy(original_cfg), hyperparameter=None)
                        metrics, times = run_pipeline(cfg)
                        method_to_metrics[method].append({**metrics, **times})
                        
                        # We force restart in some methods so we avoid forced restart for other methods
                        cfg._force_restart = ""
                        cfg._is_running_mc_dropout = False
                        
            # Save metrics to disk for future analysis and plots
            out_dir = cfg.detection.evaluation.node_evaluation._uncertainty_exp_dir
            os.makedirs(out_dir, exist_ok=True)
            method_to_metrics_path = os.path.join(out_dir, "method_to_metrics.pkl")
            torch.save(method_to_metrics, method_to_metrics_path)
            wandb.save(method_to_metrics_path, out_dir)
            
            averaged_metrics = avg_std_metrics(method_to_metrics)
            wandb.log(averaged_metrics)
            
        else:
            raise ValueError(f"Invalid experiment {cfg.experiment.used_method}")

    
    log("==" * 30)
    log("Run finished.")
    log("==" * 30)
    


if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    
    exp_name = args.exp if args.exp != "" else f"{args.dataset}_{args.model}"
        # "|".join([f"{k.split('.')[-1]}={v}" for k, v in args.__dict__.items() if "." in k and v is not None])
    tags = args.tags.split(",") if args.tags != "" else [args.model]
    
    PROJECT_PREFIX = "framework_"
    if args.project != "":
        project = args.project
    elif args.experiment == "uncertainty":
        project = "uncertainty"
    elif args.experiment == "run_n_times":
        project = f"component_ablation_study_{args.model}"
    else:
        project = "project_name"
    
    wandb.init(
        mode="online" if (args.wandb and args.tuning_mode == "none") else "disabled",
        project=PROJECT_PREFIX + project,
        name=exp_name,
        tags=tags,
    )
    
    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg)
    
    wandb.finish()
    
    # If it's a one-time run, we delete the files as we can't leverage them in future
    if cfg._restart_from_scratch:
        shutil.rmtree(cfg.preprocessing.build_graphs._task_path, ignore_errors=True)
