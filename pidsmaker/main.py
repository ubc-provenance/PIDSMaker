"""PIDSMaker main entry point.

This module orchestrates the complete PIDS pipeline execution across three modes:
1. Standard mode: Single run through all pipeline tasks
2. Uncertainty mode: Multiple runs for uncertainty quantification (MC Dropout, deep ensemble, hyperparameter sensitivity)
3. Tuning mode: Hyperparameter sweeps using Weights & Biases

The pipeline executes 8 tasks sequentially: construction → transformation →
featurization → feat_inference → batching → training → evaluation → triage
"""

import argparse
import copy
import os
import shutil
import time
from collections import defaultdict

import torch
import wandb

from pidsmaker.config import (
    get_runtime_required_args,
    get_uncertainty_methods_to_run,
    get_yml_cfg,
    set_task_to_done,
    update_task_paths_to_restart,
)
from pidsmaker.experiments.tuning import (
    fuse_cfg_with_sweep_cfg,
    get_tuning_sweep_cfg,
)
from pidsmaker.experiments.uncertainty import (
    avg_std_metrics,
    fuse_hyperparameter_metrics,
    max_metrics,
    min_metrics,
    push_best_files_to_wandb,
    update_cfg_for_uncertainty_exp,
)
from pidsmaker.tasks import (
    batching,
    construction,
    evaluation,
    feat_inference,
    featurization,
    training,
    transformation,
    triage,
)
from pidsmaker.utils.utils import log, remove_underscore_keys, set_seed


def get_task_to_module(cfg):
    """Map task names to their corresponding modules and task paths.

    Args:
        cfg: Configuration object containing task paths

    Returns:
        dict: Mapping of task names to module and task_path information
    """
    return {
        "construction": {
            "module": construction,
            "task_path": cfg.construction._task_path,
        },
        "transformation": {
            "module": transformation,
            "task_path": cfg.transformation._task_path,
        },
        "featurization": {
            "module": featurization,
            "task_path": cfg.featurization._task_path,
        },
        "feat_inference": {
            "module": feat_inference,
            "task_path": cfg.feat_inference._task_path,
        },
        "batching": {
            "module": batching,
            "task_path": cfg.batching._task_path,
        },
        "training": {
            "module": training,
            "task_path": cfg.training._task_path,
        },
        "evaluation": {
            "module": evaluation,
            "task_path": cfg.evaluation._task_path,
        },
        "triage": {
            "module": triage,
            "task_path": cfg.triage._task_path,
        },
    }


def clean_cfg_for_log(cfg):
    """Clean configuration for W&B logging by removing internal keys.

    Args:
        cfg: Configuration object

    Returns:
        dict: Cleaned configuration with underscore keys removed except specified ones
    """
    return remove_underscore_keys(
        dict(cfg), keys_to_keep=["_task_path", "_exp", "_tuning_file_path"]
    )


def main(cfg, project=None, exp=None, sweep_id=None, **kwargs):
    """Main execution function for PIDSMaker pipeline.

    Orchestrates pipeline execution across three modes: standard, uncertainty, and tuning.
    Handles task scheduling, restart logic, and metric collection.

    Args:
        cfg: Configuration object with all pipeline settings
        project: W&B project name for logging
        exp: Experiment name for tracking
        sweep_id: W&B sweep ID for tuning mode (optional)
        **kwargs: Additional keyword arguments
    """
    set_seed(cfg)

    def run_task(task: str, cfg, method=None, iteration=None):
        """Execute a single pipeline task with restart logic and timing.

        Args:
            task: Task name (construction, transformation, featurization, etc.)
            cfg: Configuration object
            method: Uncertainty method name (for deep_ensemble restart logic)
            iteration: Current iteration number (for uncertainty experiments)

        Returns:
            dict: Contains 'time' (execution time) and 'return' (task result)
        """
        start = time.time()
        return_value = None

        # We add the iteration index to subtask to have a unique folder per iteration
        if method == "deep_ensemble":
            subtask_concat_value = {
                "subtask": cfg.experiment.uncertainty.deep_ensemble.restart_from,
                "concat_value": str(iteration),
            }
        else:
            subtask_concat_value = None

        # This updates all task paths
        should_restart = update_task_paths_to_restart(
            cfg, subtask_concat_value=subtask_concat_value
        )

        task_to_module = get_task_to_module(cfg)
        module = task_to_module[task]["module"]
        task_path = task_to_module[task]["task_path"]

        if should_restart[task]:
            return_value = module.main(cfg)
            set_task_to_done(task_path)

        return {"time": time.time() - start, "return": return_value}

    def run_pipeline(cfg, method=None, iteration=None):
        """Execute complete pipeline: all 8 tasks from construction to triage.

        Args:
            cfg: Configuration object
            method: Uncertainty method name (optional, for restart logic)
            iteration: Current iteration number (optional, for deep ensemble)

        Returns:
            tuple: (metrics dict from evaluation, times dict with task execution times)
        """
        tasks = get_task_to_module(cfg).keys()
        task_results = {task: run_task(task, cfg, method, iteration) for task in tasks}

        metrics = task_results["evaluation"]["return"] or {}
        metrics = {
            **metrics,
            "val_score": task_results["training"]["return"],
        }

        times = {
            f"time_{task}": round(results["time"], 2) for task, results in task_results.items()
        }
        return metrics, times

    def run_pipeline_with_experiments(cfg):
        """Run pipeline with experiment mode handling (standard, uncertainty, or tuning).

        Execution modes:
        - Standard: Single pipeline run with metrics logged to W&B
        - Uncertainty: Multiple runs for MC Dropout, deep ensemble, or hyperparameter sensitivity
          with averaged/min/max statistics computed and logged

        Args:
            cfg: Configuration object with experiment settings
        """
        # Standard behavior: we run the whole pipeline
        if cfg.experiment.used_method == "none":
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
                    assert iterations % 2 != 0, (
                        f"The number of iterations for hyperparameters should be odd, found {iterations}"
                    )

                    hyper_to_metrics = defaultdict(list)
                    for hyper in hyperparameters:
                        log(
                            f"[@hyperparameter {hyper}] - Started",
                            pre_return_line=True,
                        )

                        for i in range(iterations):
                            log(f"[@iteration {i}]", pre_return_line=True)
                            cfg = update_cfg_for_uncertainty_exp(
                                method,
                                i,
                                iterations,
                                copy.deepcopy(original_cfg),
                                hyperparameter=hyper,
                            )
                            metrics, times = run_pipeline(cfg, method=method, iteration=i)
                            hyper_to_metrics[hyper].append({**metrics, **times})

                    metrics = fuse_hyperparameter_metrics(hyper_to_metrics)
                    method_to_metrics[method] = metrics

                else:
                    if method == "deep_ensemble":
                        new_tag = f"from_{cfg.experiment.uncertainty.deep_ensemble.restart_from}"
                        wandb.run.tags = list(wandb.run.tags) + [str(new_tag)]

                    for i in range(iterations):
                        log(f"[@iteration {i}]", pre_return_line=True)
                        cfg = update_cfg_for_uncertainty_exp(
                            method,
                            i,
                            iterations,
                            copy.deepcopy(original_cfg),
                            hyperparameter=None,
                        )
                        metrics, times = run_pipeline(cfg, method=method, iteration=i)
                        method_to_metrics[method].append({**metrics, **times})

                        # We force restart in some methods so we avoid forced restart for other methods
                        cfg._force_restart = ""
                        cfg._is_running_mc_dropout = False

            # Save metrics to disk for future analysis and plots
            out_dir = cfg.evaluation._uncertainty_exp_dir
            os.makedirs(out_dir, exist_ok=True)
            method_to_metrics_path = os.path.join(out_dir, "method_to_metrics.pkl")
            torch.save(method_to_metrics, method_to_metrics_path)
            wandb.save(method_to_metrics_path, out_dir)

            averaged_metrics = avg_std_metrics(method_to_metrics)
            minimum_metrics = min_metrics(method_to_metrics)
            maximum_metrics = max_metrics(method_to_metrics)

            wandb.log(averaged_metrics)
            wandb.log(minimum_metrics)
            wandb.log(maximum_metrics)

            push_best_files_to_wandb(method_to_metrics, cfg)

        else:
            raise ValueError(f"Invalid experiment {cfg.experiment.used_method}")

    # Normal mode
    if cfg._tuning_mode == "none":
        run_pipeline_with_experiments(cfg)

    # Sweep  mode
    else:
        log("Running pipeline in 'Tuning' mode.")
        sweep_config = get_tuning_sweep_cfg(cfg)
        if not sweep_id:
            sweep_config["name"] = exp
            sweep_id = wandb.sweep(sweep_config, project=project)
            log(f"Sweep ID: YOUR_ORG/{project}/{sweep_id}")

        def run_pipeline_from_sweep(cfg):
            """Execute pipeline for a single hyperparameter configuration from W&B sweep.

            Args:
                cfg: Base configuration object (will be updated with sweep parameters)
            """
            with wandb.init(name=exp):
                sweep_cfg = wandb.config
                cfg = fuse_cfg_with_sweep_cfg(cfg, sweep_cfg)

                wandb.run.name = exp
                wandb.run.save()
                wandb.log({"dataset": cfg.dataset.name, "exp": exp})

                run_pipeline_with_experiments(cfg)

        count = sweep_config["count"] if "count" in sweep_config else None
        wandb.agent(sweep_id, lambda: run_pipeline_from_sweep(cfg), count=count)

    log("==" * 30)
    log("Run finished.")
    log("==" * 30)


if __name__ == "__main__":
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)

    exp_name = (
        args.exp.replace("dataset", args.dataset)
        if args.exp != ""
        else f"{args.dataset}_{args.model}"
    )
    tags = args.tags.split(",") if args.tags != "" else [args.model]

    wandb.init(
        mode=("online" if (args.wandb and args.tuning_mode == "none") else "disabled"),
        project=args.project,
        name=exp_name,
        tags=tags,
    )

    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(clean_cfg_for_log(cfg))

    main(cfg, project=args.project, exp=exp_name, sweep_id=args.sweep_id)

    wandb.finish()

    # If it's a one-time run, we delete the files as we can't leverage them in future
    if cfg._restart_from_scratch:
        shutil.rmtree(cfg.construction._task_path, ignore_errors=True)
