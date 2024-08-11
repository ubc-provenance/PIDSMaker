from provnet_utils import *
from config import *

def main(cfg):
    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    split_files = cfg.dataset.train_files
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)

    nx_g = torch.load(sorted_paths[0])
    print(nx_g)

    for u, v, key, attr in nx_g.edges(data=True, keys=True):
        print(f"Edge from {u} to {v} with key {key} has attributes {attr}")
        print(type(u), type(v), type(key), type(attr))

    # for node, attr in nx_g.nodes(data=True):
    #     print(f"Node {node} has attributes {attr}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)