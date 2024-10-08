from collections import defaultdict

from config import *
from provnet_utils import *
from .transformation_methods import (
    transformation_rcaid_pseudo_graph,
)

def apply_transformations(graph, methods, cfg):
    for method in methods:
        if method == "none":
            pass
        elif method == "rcaid_pseudo_graph":
            graph = transformation_rcaid_pseudo_graph.main(graph, cfg)
        else:
            raise ValueError(f"Unrecognized transformation method: {method}")

    return graph

def main(cfg):
    methods = cfg.preprocessing.transformation.used_methods
    methods = list(map(lambda x: x.strip(), methods.split(",")))
    
    # If no transformation is used, we copy all original graphs to the transformation task path
    if len(methods) == 1 and methods[0] == "none":
        src = cfg.preprocessing.build_graphs._graphs_dir
        dst = cfg.preprocessing.transformation._graphs_dir
        copy_directory(src, dst)

    else:
        base_dir = cfg.preprocessing.build_graphs._graphs_dir
        dst_dir = cfg.preprocessing.transformation._graphs_dir
        graph_list = defaultdict(list)

        split_to_files = {
            "train": get_all_files_from_folders(base_dir, cfg.dataset.train_files),
            "val": get_all_files_from_folders(base_dir, cfg.dataset.val_files),
            "test": get_all_files_from_folders(base_dir, cfg.dataset.test_files),
        }
        for split, files in split_to_files.items():
            for path in tqdm(files, desc=f'Transforming ({split})'):
                graph = torch.load(path)
                
                # Apply all transformations to a single graph
                graph = apply_transformations(graph, methods, cfg)
                
                graph_list[split].append(graph)
            
        # We save to disk at the very end to avoid errors once a file is replaced on disk
        for split, files in split_to_files.items():
            for g, path in zip(graph_list[split], files):
                file_name = path.split("/")[-1]
                dst_path = os.path.join(dst_dir, file_name)
                log(f"Creating file '{file_name}'...")
                torch.save(g, path)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
