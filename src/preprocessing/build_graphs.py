from config import *
from provnet_utils import get_multi_datasets
from .build_graph_methods import (
    build_orthrus_graphs,
    build_magic_graphs,
)
            
def main_from_config(cfg):    
    graph_method  = cfg.preprocessing.build_graphs.used_method
    if graph_method == "orthrus":
        build_orthrus_graphs.main(cfg)
    elif graph_method == "magic":
        build_orthrus_graphs.main(cfg)
        build_magic_graphs.main(cfg)
    else:
        raise ValueError(f"Unrecognized graph method: {graph_method}")

def main(cfg):
    multi_datasets = get_multi_datasets(cfg)
    if "none" in multi_datasets:
        main_from_config(cfg)
    
    # Multi-dataset mode
    else:
        for dataset in multi_datasets:
            updated_cfg, should_restart = update_cfg_for_multi_dataset(cfg, dataset)
            
            if should_restart["build_graphs"]:
                main_from_config(updated_cfg)
