"""PIDSMaker main entry point.

This module orchestrates the complete PIDS pipeline execution across three modes:
1. Standard mode: Single run through all pipeline tasks
2. Uncertainty mode: Multiple runs for uncertainty quantification (MC Dropout, deep ensemble, hyperparameter sensitivity)
3. Tuning mode: Hyperparameter sweeps using Weights & Biases

The pipeline executes 8 tasks sequentially: construction → transformation →
featurization → feat_inference → batching → training → evaluation → triage
"""

# First: sets the CPU threads before torch is imported (see pidsmaker/__init__.py)
import pidsmaker  # noqa: F401

# isort: split
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
    summarize_task_args,
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
    prepare_for_deep_ensemble,
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
from pidsmaker.utils import console
from pidsmaker.utils.loss_files import prune_non_best_epochs
from pidsmaker.utils.utils import (
    get_device,
    log,
    remove_underscore_keys,
    set_seed,
    warn_if_low_memory,
)


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


def _home(path):
    home = os.path.expanduser("~")
    path = os.path.abspath(path)
    return "~" + path[len(home) :] if path == home or path.startswith(home + os.sep) else path


def _dataset_dir(cfg):
    """Folder of the dataset of `cfg`."""
    try:
        from pidsmaker.datasets import get_datasets_dir
        from pidsmaker.utils.database import database_name

        datasets_dir = get_datasets_dir(getattr(cfg.database, "datasets_dir", None))
        return _home(os.path.join(datasets_dir, database_name(cfg)))
    except Exception:
        return None


def log_run_header(cfg):
    """Logs what runs, on which device, and where its files are."""
    try:
        from importlib.metadata import version

        pidsmaker_version = f" {version('pidsmaker')}"
    except Exception:
        pidsmaker_version = ""

    device = get_device(cfg)
    if device.type == "cuda":
        device = f"cuda ({torch.cuda.get_device_name(device)})"

    mode = cfg.experiment.used_method
    if cfg._tuning_mode != "none":
        mode = f"tuning ({cfg._tuning_mode})"

    wandb_url = None
    try:
        wandb_url = wandb.run.url if wandb.run is not None and not wandb.run.disabled else None
    except Exception:
        pass

    rows = [
        ("dataset", _dataset_dir(cfg)),
        ("artifacts", _home(cfg._artifact_dir)),
        ("device", str(device)),
        ("mode", mode if mode != "none" else None),
        ("wandb", wandb_url),
    ]
    log(
        console.style(
            f"PIDSMaker{pidsmaker_version}  {console.DOT}  {cfg._model} on {cfg.dataset.name}",
            "bold",
        )
    )
    for name, value in rows:
        if value:
            log(f"  {console.style(name.ljust(9), 'dim')}  {value}")


def download_dataset_if_needed(cfg):
    """Downloads the dataset at the start of the run if it is missing and a task runs, rather
    than in the middle of the run when a task first reads it (e.g. the labels of the evaluation,
    after the training)."""
    from pidsmaker.datasets import ensure_downloaded

    if os.environ.get("PIDSMAKER_AUTO_DOWNLOAD", "1") == "0":
        return  # The task that reads the dataset raises an error with the command to download it
    if any(update_task_paths_to_restart(cfg).values()):
        ensure_downloaded(cfg)


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
    run_start = time.time()
    console.set_artifact_dir(cfg._artifact_dir)
    download_dataset_if_needed(cfg)
    log_run_header(cfg)
    set_seed(cfg)
    warn_if_low_memory()

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

        # Deep ensemble mode modifies cfg so that it restarts some tasks
        if method == "deep_ensemble":
            should_restart, cfg = prepare_for_deep_ensemble(cfg, iteration)

        # This updates all task paths if needed
        else:
            should_restart = update_task_paths_to_restart(cfg)

        task_to_module = get_task_to_module(cfg)
        module = task_to_module[task]["module"]
        task_path = task_to_module[task]["task_path"]

        # In uncertainty experiments, the metrics of each iteration are required, so the
        # evaluation is computed again even if the trained model is already on disk
        rerun_for_metrics = method is not None and task == "evaluation"
        ran = bool(should_restart[task] or rerun_for_metrics)
        if ran:
            console.task_start(task, task_path, summarize_task_args(cfg, task))
            return_value = module.main(cfg)
            set_task_to_done(task_path)
            console.task_end(task, time.time() - start)
        else:
            console.task_cached(task, task_path)

        return {"time": time.time() - start, "return": return_value, "ran": ran}

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
        ran = [(task, r["time"]) for task, r in task_results.items() if r["ran"]]
        if ran:
            print(flush=True)
            console.table(
                ["task", "time"],
                [(task, console.format_duration(t)) for task, t in ran]
                + [("total", console.format_duration(sum(t for _, t in ran)))],
            )

        if getattr(cfg, "_keep_best_epoch_only", False):
            prune_non_best_epochs(cfg, metrics)
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
            metrics, times = run_pipeline(cfg)
            wandb.log(metrics)
            wandb.log(times)

        elif cfg.experiment.used_method == "uncertainty":
            method_to_metrics = defaultdict(list)
            original_cfg = copy.deepcopy(cfg)

            uncertainty_methods = get_uncertainty_methods_to_run(cfg)
            for method in uncertainty_methods:
                iterations = getattr(cfg.experiment.uncertainty, method).iterations
                console.section(f"uncertainty {console.DOT} {method}")

                if method == "hyperparameter":
                    hyperparameters = cfg.experiment.uncertainty.hyperparameter.hyperparameters
                    hyperparameters = map(lambda x: x.strip(), hyperparameters.split(","))
                    assert iterations % 2 != 0, (
                        f"The number of iterations for hyperparameters should be odd, found {iterations}"
                    )

                    hyper_to_metrics = defaultdict(list)
                    for hyper in hyperparameters:
                        console.section(f"{method} {console.DOT} {hyper}")

                        for i in range(iterations):
                            console.section(
                                f"{method} {console.DOT} {hyper} {console.DOT} "
                                f"iteration {i + 1}/{iterations}"
                            )
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
                        console.section(f"{method} {console.DOT} iteration {i + 1}/{iterations}")
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
        sweep_config = get_tuning_sweep_cfg(cfg)
        if not sweep_id:
            sweep_config["name"] = exp
            sweep_id = wandb.sweep(sweep_config, project=project)

        def run_pipeline_from_sweep(cfg):
            """Execute pipeline for a single hyperparameter configuration from W&B sweep.

            Args:
                cfg: Base configuration object (will be updated with sweep parameters)
            """
            with wandb.init(name=exp):
                sweep_cfg = wandb.config
                cfg = fuse_cfg_with_sweep_cfg(cfg, sweep_cfg)

                wandb.log({"dataset": cfg.dataset.name, "exp": exp})

                run_pipeline_with_experiments(cfg)

        count = sweep_config["count"] if "count" in sweep_config else None
        wandb.agent(sweep_id, lambda: run_pipeline_from_sweep(cfg), count=count)

    print(flush=True)
    log(
        console.style(
            f"{console.CHECK} Run finished in {console.format_duration(time.time() - run_start)}",
            "bold",
            "green",
        )
    )
    if cfg._tuning_mode == "none":
        console.log(
            f"  {console.style('results'.ljust(9), 'dim')}  {cfg.evaluation._task_path}",
            shorten=False,
        )


def cli(argv=None):
    """Runs the pipeline from the command line: `pidsmaker <model> <dataset> [args]` (see
    pidsmaker/cli.py), or `python pidsmaker/main.py <model> <dataset> [args]`."""
    args, unknown_args = get_runtime_required_args(return_unknown_args=True, args=argv)

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

    # Persist the resolved config next to the run so the viz Run Browser can show it.
    try:
        run_dir = os.path.dirname(cfg.evaluation._task_path)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "run_config.yml"), "w") as f:
            f.write(cfg.dump())
    except Exception as e:
        log(f"Warning: could not save run_config.yml: {e}")

    wandb.finish()

    # If it's a one-time run, we delete the files as we can't leverage them in future
    if cfg._restart_from_scratch:
        shutil.rmtree(cfg.construction._task_path, ignore_errors=True)


if __name__ == "__main__":
    cli()
