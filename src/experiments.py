import math
import torch.nn as nn

from encoders import *

def update_cfg_for_uncertainty_exp(method: str, index: int, iterations: int, cfg, hyperparameter=None):
    index = index + 1
    
    if method == "hyperparameter":
        delta = cfg.experiments.experiment.uncertainty.hyperparameter.delta
        mid_value = math.floor(iterations / 2) + 1
        if index == mid_value:
            return cfg
        
        delta = delta * index
        if index < mid_value:
            delta = -delta
            
        if hyperparameter == "lr":
            cfg.detection.gnn_training.lr *= delta
        elif hyperparameter == "num_epochs":
            cfg.detection.gnn_training.num_epochs *= delta
        elif hyperparameter == "text_h_dim":
            cfg.featurization.embed_nodes.emb_dim *= delta
        elif hyperparameter == "gnn_h_dim":
            cfg.detection.gnn_training.node_hid_dim *= delta
        else:
            raise ValueError(f"Invalid hyperparameter {hyperparameter}")
        
    elif method == "mc_dropout":
        cfg._is_running_mc_dropout = True
        cfg._force_restart = "gnn_training"
    
    elif method == "deep_ensemble":
        cfg._force_restart = "gnn_training"
        
    elif method == "bagged_ensemble":
        min_num_days = cfg.experiments.experiment.uncertainty.bagged_ensemble.min_num_days
        num_days = min_num_days + index - 1
        available_train_days = sorted(cfg.dataset.train_files + cfg.dataset.unused_files)
        days = available_train_days[:num_days]
        cfg.dataset.train_files = days
        
    return cfg

    
def fuse_hyperparameter_metrics(hyper_to_metrics):
    pass

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
