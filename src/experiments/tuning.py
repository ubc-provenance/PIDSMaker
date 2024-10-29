import copy
import os
import functools
import yaml

from config import get_yml_file, validate_yml_file, TASK_ARGS

def get_tuning_sweep_cfg(cfg):
    if cfg._tuning_mode == "hyperparameters":
        yml_file = get_yml_file(filename=f"tuning_{cfg._model}", folder="experiments/tuning/systems/")
    elif cfg._tuning_mode == "best_featurization_methods":
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
            method = "_".join(value.split("_")[:-1])
            split = value.split("_")[-1]
            if split not in ["train", "all"]:
                raise ValueError(f"Invalid split {split} for embedding technique")
            
            yml_file = get_yml_file(method, folder="tuned_components/")
            validate_yml_file(yml_file, TASK_ARGS)
            cfg.merge_from_file(yml_file)
            
            # Train or Train+Test embedding method training
            cfg.featurization.embed_nodes.training_split = split
            
        # default cfg path
        else:
            set_nested_attr(cfg, key, value)
    return cfg
