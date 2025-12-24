import os

import torch

from pidsmaker.utils.utils import log_start


def main(cfg):
    log_start(__file__)

    # TODO: this part is broken, need some work to make it work (the training part of this method seems too complicated)

    trained_w2v_dir = cfg.feat_training.alacarte._vec_graphs_dir
    indexid2vec = torch.load(os.path.join(trained_w2v_dir, "indexid2vec"))

    return indexid2vec
