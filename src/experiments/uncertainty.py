import os
import shutil
import math
import torch.nn as nn
import numpy as np
import wandb
from torch_geometric.nn import MessagePassing


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
        restart_from = cfg.experiment.uncertainty.deep_ensemble.restart_from
        if restart_from == "embed_nodes":
            clear_files_from_embed_nodes(cfg)
        elif restart_from == "gnn_training":
            clear_files_from_gnn_training(cfg)
        else:
            raise ValueError(f"Unsupported 'restart from' value: {restart_from}")
        
    elif method == "bagged_ensemble":
        # Here, force_restart will be at the beninning so no need to rm files
        min_num_days = cfg.experiment.uncertainty.bagged_ensemble.min_num_days
        num_days = min_num_days + index - 1
        available_train_days = sorted(cfg.dataset.train_files + cfg.dataset.unused_files)
        days = available_train_days[:num_days]
        cfg.dataset.train_files = days
        
    return cfg


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
    
def include_metric_in_stats(value):
    return np.isreal(value) and not isinstance(value, wandb.Image)
    
def fuse_hyperparameter_metrics(method_to_metrics):
    """
    For each hyperparameter i \in H at iteration j, we do the mean of the metrics for all hyperparameters
    in H. Basically, we compute a single list of metrics from all lists of metrics for each hyperparam.
    """
    mean_metrics = {}
    metrics = method_to_metrics[list(method_to_metrics.keys())[0]][0].items()

    for metric, val in metrics:
        if include_metric_in_stats(val):
            all_values = []
            for param, list_of_dict in method_to_metrics.items():
                values = [d[metric] for d in list_of_dict if "precision" in d]
                all_values.append(values)
            mean_metrics[metric] = np.mean(all_values, axis=0)

    list_of_dict = [dict(zip(mean_metrics.keys(), values)) for values in zip(*mean_metrics.values())]
    return list_of_dict

def avg_std_metrics(method_to_metrics):
    metrics = fuse_hyperparameter_metrics(method_to_metrics)
    
    result = {}
    metric_keys = metrics[0].keys()
    for key in metric_keys:
        values = [entry[key] for entry in metrics]
        result[f"{key}_mean"] = np.mean(values)
        result[f"{key}_std"] = np.std(values)
        result[f"{key}_std_rel"] = np.std(values) / (np.mean(values) + 1e-12) * 100
    
    return result

def max_metrics(method_to_metrics, metric='adp_score'):
    metrics = method_to_metrics[list(method_to_metrics.keys())[0]]
    max_idx = np.argmax([m[metric] for m in metrics])
    
    result = {}
    metric_keys = metrics[0].keys()
    for key in metric_keys:
        value = metrics[max_idx][key]
        if include_metric_in_stats(value):
            result[f"{key}_max"] = value
    
    return result

def min_metrics(method_to_metrics, metric='adp_score'):
    metrics = method_to_metrics[list(method_to_metrics.keys())[0]]
    min_idx = np.argmin([m[metric] for m in metrics])
    
    result = {}
    metric_keys = metrics[0].keys()
    for key in metric_keys:
        value = metrics[min_idx][key]
        if include_metric_in_stats(value):
            result[f"{key}_min"] = value
    
    return result

def push_best_files_to_wandb(method_to_metrics, cfg):
    if "deep_ensemble" in method_to_metrics:
        best_run = best_metric_pick_best_run(method_to_metrics)
        for metric, value in best_run.items():
            if metric.endswith("img"):
                out_dir = "/".join(best_run["scores_file"].split("/")[:-1])
                wandb.save(best_run["scores_file"], out_dir) # saves the scores for the best run
        wandb.log(best_run) # logs all best metrics and images for easy analysis

def best_metric_pick_best_run(method_to_metrics):
    metrics = method_to_metrics["deep_ensemble"]
    adp_scores = np.array([e["adp_score"] for e in metrics])
    max_adp_mask = adp_scores == adp_scores.max()
    
    # Filter only the elements with max adp_score and get the one with the highest discrimination
    filtered_metrics = [metrics[i] for i in range(len(metrics)) if max_adp_mask[i]]
    best_run = max(filtered_metrics, key=lambda e: e["discrimination"])
    
    return best_run

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
