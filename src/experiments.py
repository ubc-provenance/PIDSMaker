import math

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
            cfg.detection.gnn_training.lr = cfg.detection.gnn_training.lr * delta
        elif hyperparameter == "num_epochs":
            cfg.detection.gnn_training.num_epochs = cfg.detection.gnn_training.num_epochs * delta
        elif hyperparameter == "text_h_dim":
            cfg.featurization.embed_nodes.emb_dim = cfg.featurization.embed_nodes.emb_dim * delta
        elif hyperparameter == "gnn_h_dim":
            cfg.detection.gnn_training.node_hid_dim = cfg.detection.gnn_training.node_hid_dim * delta
        else:
            raise ValueError(f"Invalid hyperparameter {hyperparameter}")
        
    return cfg

    
def fuse_hyperparameter_metrics(hyper_to_metrics):
    pass