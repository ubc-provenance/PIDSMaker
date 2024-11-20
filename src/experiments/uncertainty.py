import math
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import wandb
from collections import defaultdict
from pprint import pprint

from encoders import *

def update_cfg_for_uncertainty_exp(method: str, index: int, iterations: int, cfg, hyperparameter=None):
    index = index + 1
    
    if method == "hyperparameter":
        delta = cfg.experiment.uncertainty.hyperparameter.delta
        mid_value = math.floor(iterations / 2) + 1
        
        # we want delta to be like [-0.4, -0.2, 0, 0.2, 0.4], for delta=0.2
        if index < mid_value:
            delta = -delta * (mid_value - index)
        elif index > mid_value:
            delta = delta * index
        else:
            delta = 0
            
        if hyperparameter == "text_h_dim":
            clear_files_from_embed_nodes(cfg)
            if cfg.featurization.embed_nodes.emb_dim is not None:
                cfg.featurization.embed_nodes.emb_dim += int(delta * cfg.featurization.embed_nodes.emb_dim)
        else:
            clear_files_from_gnn_training(cfg)
            if hyperparameter == "lr":
                cfg.detection.gnn_training.lr += delta * cfg.detection.gnn_training.lr
            elif hyperparameter == "num_epochs":
                cfg.detection.gnn_training.num_epochs += int(delta * cfg.detection.gnn_training.num_epochs)
            elif hyperparameter == "gnn_h_dim":
                cfg.detection.gnn_training.node_hid_dim += int(delta * cfg.detection.gnn_training.node_hid_dim)
            else:
                raise ValueError(f"Invalid hyperparameter {hyperparameter}")
        
    elif method == "mc_dropout":
        clear_files_from_gnn_training(cfg)
        cfg._is_running_mc_dropout = True
    
    elif method == "deep_ensemble":
        clear_files_from_gnn_training(cfg)
        
    elif method == "bagged_ensemble":
        # Here, force_restart will be at the beninning so no need to rm files
        min_num_days = cfg.experiment.uncertainty.bagged_ensemble.min_num_days
        num_days = min_num_days + index - 1
        available_train_days = sorted(cfg.dataset.train_files + cfg.dataset.unused_files)
        days = available_train_days[:num_days]
        cfg.dataset.train_files = days
        
    return cfg

def plot_metric(metric_to_plots: list[str], method_to_metrics, cfg):
    # Base style definitions for methods
    method_to_style = {
        "hyperparameter": {
            "label": "Hyperparameter Ensemble",
            "marker": "o",
            "linestyle": "-",
            "color": '#1f77b4',
        },
        "mc_dropout": {
            "label": "MC Dropout",
            "marker": "s",
            "linestyle": "--",
            "color": '#ff7f0e',
        },
        "deep_ensemble": {
            "label": "Deep Ensemble",
            "marker": "D",
            "linestyle": "-",
            "color": '#2ca02c',
        },
        "bagged_ensemble": {
            "label": "Bagged Ensemble",
            "marker": "x",
            "linestyle": ":",
            "color": '#d62728',
        },
    }
    
    plt.figure(figsize=(8, 6))
    longest_x_axis = []

    # Differentiate the curves for multiple metrics by applying varying styles
    for i, metric_to_plot in enumerate(metric_to_plots):
        # Modify the linestyle and marker for each metric differently
        metric_suffix_styles = {
            0: {"linestyle": "solid", "marker": "s"},   # Metric 1 (solid line)
            1: {"linestyle": "dashed", "marker": "x"},  # Metric 2 (dashed line)
        }

        # Loop through each method and plot metrics
        for method, metrics in method_to_metrics.items():
            y = [m[metric_to_plot] for m in metrics]
            x = list(range(1, len(y) + 1))

            # Adjust x-axis for longest series
            if len(x) > len(longest_x_axis):
                longest_x_axis = x

            # Retrieve base style and customize for each metric using the suffix style
            style = method_to_style[method]
            suffix_style = metric_suffix_styles[i % len(metric_suffix_styles)]  # Cycle through suffix styles

            metric_label = metric_to_plot.upper() if metric_to_plot in ["ap", "mcc"] else metric_to_plot
            plt.plot(
                x, y, label=f"{style['label']} ({metric_label})",  # Add metric to the label for clarity
                marker=style["marker"] if suffix_style["marker"] is None else suffix_style["marker"],
                linestyle=suffix_style["linestyle"],
                color=style["color"],
                linewidth=2  # You can adjust line width for better visibility
            )

            log(f"Metrics for {metric_to_plot}: {method}")
            log(y)

    metric_to_plot = metric_to_plots[0]
    if metric_to_plot == "ap":
        plt.yscale('log')

    if metric_to_plot == "val_ap":
        plt.yticks([0.0, 1.0])
    else:
        plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
        
    # plt.xlabel("Iteration", fontsize=14)
    # plt.ylabel("Metric", fontsize=14)

    plt.xticks(longest_x_axis)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_dir = cfg.detection.evaluation.node_evaluation._uncertainty_exp_dir
    os.makedirs(out_dir, exist_ok=True)
    log(f"Saving uncertainty figures to {out_dir}...")
    
    # Without legend
    plot1 = os.path.join(out_dir, f"uncertainty_{metric_to_plot}_{cfg._model}")
    plt.savefig(plot1+".png")
    plt.savefig(plot1+".svg")
    wandb.save(plot1+".svg", out_dir)

    # With legend
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    plot2 = os.path.join(out_dir, f"uncertainty_{metric_to_plot}_with_legend_{cfg._model}")
    plt.savefig(plot2+".png")
    plt.savefig(plot2+".svg")
    wandb.save(plot2+".svg", out_dir)

    return {
        f"uncertainty_{metric_to_plot}": wandb.Image(plot1+".png"),
        f"uncertainty_with_legend_{metric_to_plot}": wandb.Image(plot2+".png"),
    }

PICKED_METRICS = ["ap", "precision", "val_ap", "mcc", "adp_score"]
def compute_uncertainty_stats(method_to_metrics):
    pprint(method_to_metrics)
    
    stats = defaultdict(dict)
    # all_plot_imgs = {}
    
    for met in PICKED_METRICS:
        # metrics_to_plot = ["precision"] if met == "precision" else [met, "precision"] # we plot both a metric and precision
        # plots_imgs = plot_metric(metrics_to_plot, method_to_metrics, cfg)
        # # all_plot_imgs = {**all_plot_imgs, **plots_imgs}
        
        for method, metrics in method_to_metrics.items():
        
            values = [m[met] for m in metrics]
            
            stats[method][f"{met}_std"] = round(np.std(values), 6)
            stats[method][f"{met}_%std"] = round(np.std(values) / np.mean(values), 6)
            stats[method][f"{met}_%relative_range"] = round((np.max(values) - np.min(values)) / np.mean(values), 6)
            stats[method][f"{met}_variance"] = round(np.var(values), 6)
            stats[method][f"{met}_mean"] = round(np.mean(values), 6)
            stats[method][f"{met}_max-mean"] = round(np.max(values) - np.mean(values), 6)
            stats[method][f"{met}_mean-min"] = round(np.mean(values) - np.min(values), 6)
            stats[method][f"{met}_monotonicity"] = monotonicity_metric(values)
            
    stats = {
        "uncertainty": {
            **stats,
            # **all_plot_imgs,
        }
    }
    return stats


# Utils
def clear_files_from_gnn_training(cfg):
    """Removes task paths to avoid old artifacts + consequently force restarts from gnn_training"""
    paths = [cfg.detection.gnn_training._task_path, cfg.detection.evaluation._task_path]
    
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
    
def clear_files_from_embed_nodes(cfg):
    paths = [cfg.featurization.embed_nodes._task_path, cfg.featurization.embed_edges._task_path]
    
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
        
    clear_files_from_gnn_training(cfg)
    
def fuse_hyperparameter_metrics(hyper_to_metrics):
    """
    For each hyperparameter i \in H at iteration j, we do the mean of the metrics for all hyperparameters
    in H. Basically, we compute a single list of metrics from all lists of metrics for each hyperparam.
    """
    mean_metrics = {}
    metrics = hyper_to_metrics[list(hyper_to_metrics.keys())[0]][0].keys()

    for metric in metrics:
        if metric in PICKED_METRICS:
            all_values = []
            for param, list_of_dict in hyper_to_metrics.items():
                values = [d[metric] for d in list_of_dict]
                all_values.append(values)
            mean_metrics[metric] = np.mean(all_values, axis=0)

    list_of_dict = [dict(zip(mean_metrics.keys(), values)) for values in zip(*mean_metrics.values())]
    return list_of_dict

def monotonicity_metric(values):
    """
    Calculate the monotonicity metric, which measures the fraction of pairs in a sequence that are increasing.
    
    :param values: List or array of numerical values.
    :return: Monotonicity metric (value between 0 and 1).
    """
    if len(values) < 2:
        return 1  # If there are less than 2 values, we consider it to be monotonic
    
    increasing_count = sum(1 for i in range(len(values) - 1) if values[i + 1] > values[i])
    total_pairs = len(values) - 1
    
    return increasing_count / total_pairs if total_pairs > 0 else 1

# MC Dropout
class DropoutWrapper(nn.Module):
    def __init__(self, module: nn.Module, p):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p)
    
    def forward(self, x, *args, **kwargs):
        x = self.dropout(x)
        x = self.module(x, *args, **kwargs)
        return x
    
class IdentityWrapper(nn.Module):    
    def forward(self, x, *args, **kwargs):
        return x
    
LAYERS_WITH_DROPOUT = [nn.Linear, nn.GRU, MessagePassing]
def add_dropout_to_model(model, p, layers_with_dropout=LAYERS_WITH_DROPOUT):
    """Add Dropout layers after each Linear layer, except the last one."""
    
    def add_dropout_to_modules(model):
        modules = list(model.named_children())

        for name, module in modules:
            if isinstance(module, nn.Dropout): # we remove all initial dropout
                setattr(model, name, IdentityWrapper())
            elif any(isinstance(module, l) for l in layers_with_dropout): # and add dropout before layers
                setattr(model, name, DropoutWrapper(module, p=p))
            else:
                add_dropout_to_modules(module)
        return model

    add_dropout_to_modules(model.encoder)
    
def activate_dropout_inference(model):
    """Enable dropout during inference (MC Dropout), including nested modules."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train() 
