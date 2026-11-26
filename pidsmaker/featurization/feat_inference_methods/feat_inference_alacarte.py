import os

import torch



def main(cfg):

    # TODO: this part is broken, need some work to make it work (the training part of this method seems too complicated)

    trained_w2v_dir = cfg.featurization.alacarte._vec_graphs_dir
    indexid2vec = torch.load(os.path.join(trained_w2v_dir, "indexid2vec"))

    return indexid2vec
