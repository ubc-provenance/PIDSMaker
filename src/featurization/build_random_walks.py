from typing import Literal
from config import *
from provnet_utils import *
import os
import torch


def preprocess_split(split: Literal["train", "val", "test"], split_files: list[str], cfg):
    # Concatenates all files from all folders from this set and flattens in a single list
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    num_walks = cfg.featurization.embed_nodes.word2vec.num_walks

    sorted_paths = get_all_files_from_folders(base_dir, split_files)
    
    g = []
    graph_info = open(f"{cfg.featurization.embed_nodes.word2vec._random_walk_dir}/graph_info.csv", "w")
    writer = csv.writer(graph_info)
    random_walks_file = os.path.join(cfg.featurization.embed_nodes.word2vec._random_walk_corpus_dir, f"{split}.csv")
    random_walks_file_fd = open(random_walks_file, 'w')
    adjacency_path = os.path.join(cfg.featurization.embed_nodes.word2vec._random_walk_dir, f"{split}-adj")
    
    for path in sorted_paths:
        file = path.split("/")[-1]
        adjacency_file = os.path.join(adjacency_path, f"{split}-{file}.csv")
        os.makedirs(adjacency_path, exist_ok=True)

        print("load file: ", path)
        graph = torch.load(path)
        gen_darpa_adj_files(graph, adjacency_file)
        g.append(graph)
        writer.writerow([
            adjacency_file,
            len(graph.nodes)
        ])
        corpus_file = os.path.join(cfg.featurization.embed_nodes.word2vec._random_walk_dir, f"{split}_set_corpus.csv")
        gen_darpa_rw_file(
            graph=graph,
            walk_len=cfg.featurization.embed_nodes.word2vec.walk_length,
            filename=corpus_file,
            adjfilename=adjacency_file,
            overall_fd=random_walks_file_fd,
            num_walks=num_walks,
        )
    graph_info.close()
    random_walks_file_fd.close()

def main(cfg):
    os.makedirs(cfg.featurization.embed_nodes.word2vec._random_walk_dir, exist_ok=True)
    os.makedirs(cfg.featurization.embed_nodes.word2vec._random_walk_corpus_dir, exist_ok=True)

    preprocess_split(split="train", split_files=cfg.dataset.train_files, cfg=cfg)
    preprocess_split(split="val", split_files=cfg.dataset.val_files, cfg=cfg)
    preprocess_split(split="test", split_files=cfg.dataset.test_files, cfg=cfg)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
