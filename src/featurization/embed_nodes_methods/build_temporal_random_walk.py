import os
from provnet_utils import *
from config import *
from tqdm import tqdm
from trw import TRW
import torch

def run_temporal_random_walk(split_files, out_dir, logger, cfg):
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)

    os.makedirs(out_dir, exist_ok=True)
    walk_length = cfg.featurization.embed_nodes.temporal_rw.walk_length
    num_walks = cfg.featurization.embed_nodes.temporal_rw.num_walks
    workers = cfg.featurization.embed_nodes.temporal_rw.trw_workers
    time_weight = cfg.featurization.embed_nodes.temporal_rw.time_weight
    half_life = cfg.featurization.embed_nodes.temporal_rw.half_life

    for path in tqdm(sorted_paths, desc='Building temporal random walks'):
        file = path.split('/')[-1]

        logger.info(f'Processing {file}')
        graph = torch.load(path)

        save_path = os.path.join(out_dir, f"{file}.txt")

        trw = TRW(graph=graph,
                  walk_length=walk_length,
                  num_walks=num_walks,
                  workers=workers,
                  path_save_dir=save_path,
                  time_weight=time_weight,
                  half_life=half_life)
        trw.run()

def main(cfg):
    logger = get_logger(
        name="build_temporal_random_walk",
        filename=os.path.join(cfg.featurization.embed_nodes._logs_dir, "temporal_random_walk.log")
    )

    os.makedirs(cfg.featurization.embed_nodes.temporal_rw._random_walk_dir, exist_ok=True)
    os.makedirs(cfg.featurization.embed_nodes.temporal_rw._random_walk_corpus_dir, exist_ok=True)

    run_temporal_random_walk(split_files=cfg.dataset.train_files,
                             out_dir=os.path.join(cfg.featurization.embed_nodes.temporal_rw._random_walk_corpus_dir,
                                                  "train/"),
                             logger=logger,
                             cfg=cfg)

    run_temporal_random_walk(split_files=cfg.dataset.val_files,
                             out_dir=os.path.join(cfg.featurization.embed_nodes.temporal_rw._random_walk_corpus_dir,
                                                  "val/"),
                             logger=logger,
                             cfg=cfg)

    run_temporal_random_walk(split_files=cfg.dataset.test_files,
                             out_dir=os.path.join(cfg.featurization.embed_nodes.temporal_rw._random_walk_corpus_dir,
                                                  "test/"),
                             logger=logger,
                             cfg=cfg)

    run_temporal_random_walk(split_files=cfg.dataset.unused_files,
                             out_dir=os.path.join(cfg.featurization.embed_nodes.temporal_rw._random_walk_corpus_dir,
                                                  "unused/"),
                             logger=logger,
                             cfg=cfg)



if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)