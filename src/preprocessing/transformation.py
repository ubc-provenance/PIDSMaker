from config import get_runtime_required_args, get_yml_cfg, SYNTHETIC_ATTACKS, update_cfg_for_multi_dataset
from provnet_utils import *
from .transformation_methods import (
    transformation_rcaid_pseudo_graph,
    transformation_undirected,
    transformation_dag,
    synthetic_attack_naive,
)


def no_transformation(base_dir, dst_dir):
    # If no transformation is used, we copy all original graphs to the transformation task path
    copy_directory(base_dir, dst_dir)


def add_synthetic_attacks(base_dir, dst_dir, cfg, method):
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    train_graphs = load_graphs_for_days(base_dir, cfg.dataset.train_files)
    val_graphs = load_graphs_for_days(base_dir, cfg.dataset.val_files)
    
    processed_graphs = apply_synthetic_attacks(train_graphs, val_graphs, cfg, method)
    
    test_graphs = load_graphs_for_days(base_dir, cfg.dataset.test_files)
    graphs = [*processed_graphs, *test_graphs]
    
    os.makedirs(dst_dir, exist_ok=True)
    
    i = 0
    days = [*cfg.dataset.train_files, *cfg.dataset.val_files, *cfg.dataset.test_files]
    for day in log_tqdm(days, desc="Saving graphs to disk"):
        sorted_paths = get_all_files_from_folders(base_dir, [day])
        for path in sorted_paths:
            graph = graphs[i]
            i += 1
            
            file_name = path.split("/")[-1]
            dst_path = os.path.join(dst_dir, day)
            os.makedirs(dst_path, exist_ok=True)
            torch.save(graph, os.path.join(dst_path, file_name))


def apply_synthetic_attacks(train_graphs, val_graphs, cfg, method):
    if method == "synthetic_attack_naive":
        return synthetic_attack_naive.main(train_graphs, val_graphs, cfg)
    
    raise ValueError(f"Invalid attack generation method {method}")


def add_graph_transformation(base_dir, dst_dir, cfg, methods):
    os.makedirs(dst_dir, exist_ok=True)

    days = [*cfg.dataset.train_files, *cfg.dataset.val_files, *cfg.dataset.test_files]
    for day in log_tqdm(days, desc=f'Transforming'):
        sorted_paths = get_all_files_from_folders(base_dir, [day])
        for path in sorted_paths:
            graph = torch.load(path)
            
            # Apply all transformations to a single graph
            graph = apply_graph_transformations(graph, methods, cfg)
            
            file_name = path.split("/")[-1]
            dst_path = os.path.join(dst_dir, day)
            os.makedirs(dst_path, exist_ok=True)
            torch.save(graph, os.path.join(dst_path, file_name))


def apply_graph_transformations(graph, methods, cfg):
    for method in methods:
        if method == "none":
            pass
        elif method == "rcaid_pseudo_graph":
            graph = transformation_rcaid_pseudo_graph.main(graph, cfg)
        elif method == "undirected":
            graph = transformation_undirected.main(graph)
        elif method == "dag":
            graph = transformation_dag.main(graph)
        else:
            raise ValueError(f"Unrecognized transformation method: {method}")

    return graph

def main_from_config(cfg):
    methods = cfg.preprocessing.transformation.used_methods
    methods = list(map(lambda x: x.strip(), methods.split(",")))
    
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    dst_dir = cfg.preprocessing.transformation._graphs_dir
    
    if len(methods) == 1 and methods[0] == "none":
        no_transformation(base_dir, dst_dir)
        
    elif len(methods) == 1 and methods[0] in SYNTHETIC_ATTACKS.keys():
        add_synthetic_attacks(base_dir, dst_dir, cfg, methods[0])

    else:
        add_graph_transformation(base_dir, dst_dir, cfg, methods)

def main(cfg):
    set_seed(cfg)
    log_start(__file__)
    
    multi_datasets = get_multi_datasets(cfg)
    if "none" in multi_datasets:
        main_from_config(cfg)
    
    # Multi-dataset mode
    else:
        for dataset in multi_datasets:
            updated_cfg, should_restart = update_cfg_for_multi_dataset(cfg, dataset)
            
            if should_restart["transformation"]:
                main_from_config(updated_cfg)
  

if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
