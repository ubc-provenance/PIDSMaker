import os

import torch

from pidsmaker.featurization.utils.trw import TRW
from pidsmaker.utils.utils import get_all_files_from_folders, log_start, log_tqdm


def run_temporal_random_walk(split_files, out_dir, cfg):
    base_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)

    os.makedirs(out_dir, exist_ok=True)
    walk_length = cfg.featurization.feat_training.temporal_rw.walk_length
    num_walks = cfg.featurization.feat_training.temporal_rw.num_walks
    workers = cfg.featurization.feat_training.temporal_rw.trw_workers
    time_weight = cfg.featurization.feat_training.temporal_rw.time_weight
    half_life = cfg.featurization.feat_training.temporal_rw.half_life

    for path in log_tqdm(sorted_paths, desc="Building temporal random walks"):
        file = path.split("/")[-1]

        graph = torch.load(path)

        save_path = os.path.join(out_dir, f"{file}.txt")

        trw = TRW(
            graph=graph,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            path_save_dir=save_path,
            time_weight=time_weight,
            half_life=half_life,
        )
        trw.run()


def main(cfg):
    log_start(__file__)

    os.makedirs(cfg.featurization.feat_training.temporal_rw._random_walk_dir, exist_ok=True)
    os.makedirs(cfg.featurization.feat_training.temporal_rw._random_walk_corpus_dir, exist_ok=True)

    run_temporal_random_walk(
        split_files=cfg.dataset.train_files,
        out_dir=os.path.join(
            cfg.featurization.feat_training.temporal_rw._random_walk_corpus_dir, "train/"
        ),
        cfg=cfg,
    )

    run_temporal_random_walk(
        split_files=cfg.dataset.val_files,
        out_dir=os.path.join(
            cfg.featurization.feat_training.temporal_rw._random_walk_corpus_dir, "val/"
        ),
        cfg=cfg,
    )

    run_temporal_random_walk(
        split_files=cfg.dataset.test_files,
        out_dir=os.path.join(
            cfg.featurization.feat_training.temporal_rw._random_walk_corpus_dir, "test/"
        ),
        cfg=cfg,
    )

    run_temporal_random_walk(
        split_files=cfg.dataset.unused_files,
        out_dir=os.path.join(
            cfg.featurization.feat_training.temporal_rw._random_walk_corpus_dir, "unused/"
        ),
        cfg=cfg,
    )
