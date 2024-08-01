from provnet_utils import *
from config import *
from featurization import(
    embed_edges
)

def get_nx_graph(t, i):
    base_dir = cfg.preprocessing.build_graphs.magic_graphs_dir

    if t == "train":
        split_files = cfg.dataset.train_files
    elif t == "test":
        split_files = cfg.dataset.test_files

    sorted_paths = get_all_files_from_folders(base_dir, split_files)
    file_path = sorted_paths[i]
    nx_graph = torch.load(file_path)
    print(file_path.split('/')[-1])
    return nx_graph

def main(cfg):

    # embed_edges.main(cfg)

    nx_graph = get_nx_graph("train", 0)




if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)