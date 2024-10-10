from collections import defaultdict

from config import *
from provnet_utils import *
from .transformation_methods import (
    transformation_rcaid_pseudo_graph,
    transformation_undirected,
    transformation_dag,
)

def apply_transformations(graph, methods, cfg):
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

def main(cfg):
    log_start(__file__)
    methods = cfg.preprocessing.transformation.used_methods
    methods = list(map(lambda x: x.strip(), methods.split(",")))
    
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    dst_dir = cfg.preprocessing.transformation._graphs_dir
    
    # If no transformation is used, we copy all original graphs to the transformation task path
    if len(methods) == 1 and methods[0] == "none":
        copy_directory(base_dir, dst_dir)

    else:
        os.makedirs(dst_dir, exist_ok=True)
        graph_list = defaultdict(list)

        days = get_days_from_cfg(cfg)
        for day in tqdm(days, desc=f'Transforming'):
            sorted_paths = get_all_files_from_folders(base_dir, [f"graph_{day}"])
            for path in sorted_paths:
                graph = torch.load(path)
                
                # Apply all transformations to a single graph
                graph = apply_transformations(graph, methods, cfg)
                
                graph_list[day].append({
                    "file": path.split("/")[-1],
                    "graph": graph,
                })
            
        # We save to disk at the very end to avoid errors once a file is replaced on disk
        for day, graphs in graph_list.items():
            for d in graphs:
                file_name = d["file"]
                log(f"Creating file '{file_name}'...")
                
                dst_path = os.path.join(dst_dir, f"graph_{day}")
                os.makedirs(dst_path, exist_ok=True)
                torch.save(d["graph"], os.path.join(dst_path, file_name))


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
