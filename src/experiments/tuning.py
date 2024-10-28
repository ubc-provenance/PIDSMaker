import copy
import os
import functools
import yaml

from config import get_yml_file

def get_tuning_sweep_cfg(cfg):
    yml_file = get_yml_file(filename=f"tuning_{cfg._model}", folder="experiments/tuning/")
    if not os.path.exists(yml_file):
        raise FileNotFoundError(f"Missing tuning yml file for model {cfg._model}")
    
    with open(yml_file, 'r') as file:
        tuning_config = yaml.safe_load(file)
    return tuning_config

def set_nested_attr(d, attr, value):
    return functools.reduce(lambda c, k: c[k] if k in c else None, attr.split('.')[:-1], d).__setitem__(attr.split('.')[-1], value)

def fuse_cfg_with_sweep_cfg(cfg, sweep_cfg):
    cfg = copy.deepcopy(cfg)
    for cfg_path, value in sweep_cfg.items():
        set_nested_attr(cfg, cfg_path, value)
    return cfg
