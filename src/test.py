from provnet_utils import *
from config import *

def main(cfg):
    split_files = cfg.dataset.train_files
    graph_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(graph_dir, split_files)

    test_graph_path = sorted_paths[0]
    test_graph = torch.load(test_graph_path)

    node, data = list(test_graph.nodes(data=True))[0]
    print(f"Node 0 is {node}, its attrs are {data}")
    print(type(node))

    u, v, k, data = list(test_graph.edges(keys=True,data=True))[0]
    print(f"edge {u} -> {v} {k}, its attrs are {data}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)