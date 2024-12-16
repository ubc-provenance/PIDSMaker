import copy
import os
import functools
import yaml

from config import get_yml_file, merge_cfg_and_check_syntax

def get_tuning_sweep_cfg(cfg):
    # Priority to a manual path.
    if cfg._tuning_file_path != "":
        splitted_path = cfg._tuning_file_path.split("/")
        path = "/".join(splitted_path[:-1])
        filename = splitted_path[-1]
        yml_file = get_yml_file(filename=filename, folder=f"experiments/tuning/{path}/")
    
    else:
        if cfg._tuning_mode == "hyperparameters":
            # First looks for a specific yml file in the dataset folder. If not present, takes the default tuning config.
            yml_file = get_yml_file(filename=f"tuning_{cfg._model}", folder=f"experiments/tuning/systems/{cfg.dataset.name.lower()}/")
            if not os.path.exists(yml_file):
                yml_file = get_yml_file(filename=f"tuning_default_baselines", folder=f"experiments/tuning/systems/default/")
        
        elif cfg._tuning_mode == "featurization":
            yml_file = get_yml_file(filename="tuning_featurization_methods", folder="experiments/tuning/components/")
        
        else:
            raise ValueError(f"Invalid tuning mode {cfg._tuning_mode}")
            
    if not os.path.exists(yml_file):
        raise FileNotFoundError("Missing tuning yml file")
    
    with open(yml_file, 'r') as file:
        tuning_config = yaml.safe_load(file)
    return tuning_config

def set_nested_attr(d, attr, value):
    return functools.reduce(lambda c, k: c[k] if k in c else None, attr.split('.')[:-1], d).__setitem__(attr.split('.')[-1], value)

def fuse_cfg_with_sweep_cfg(cfg, sweep_cfg):
    cfg = copy.deepcopy(cfg)
    for key, value in sweep_cfg.items():
        
        # special keys
        if key == "embedding_techniques":
            if value == "no_featurization":
                yml_file = get_yml_file(value, folder="tuned_components/")
                merge_cfg_and_check_syntax(cfg, yml_file)
            
            else:
                method = "_".join(value.split("_")[:-1])
                split = value.split("_")[-1]
                if split not in ["train", "all"]:
                    raise ValueError(f"Invalid split {split} for embedding technique")
                
                yml_file = get_yml_file(method, folder="tuned_components/")
                merge_cfg_and_check_syntax(cfg, yml_file)
                
                # Train or Train+Test embedding method training
                cfg.featurization.embed_nodes.training_split = split
                
                # If a model doesn't use embedding in features, we add them to benchmark
                if "node_emb" not in cfg.detection.gnn_training.encoder.node_features:
                    cfg.detection.gnn_training.encoder.node_features += ",node_emb"
            
        # default cfg path
        else:
            set_nested_attr(cfg, key, value)
            
    # Special cases
    if cfg.detection.gnn_training.node_out_dim == -1:
        cfg.detection.gnn_training.node_out_dim = cfg.detection.gnn_training.node_hid_dim
    elif cfg.detection.gnn_training.node_out_dim == -2:
        cfg.detection.gnn_training.node_out_dim = cfg.detection.gnn_training.node_hid_dim // 2
    
    return cfg
