from pidsmaker.utils.utils import set_seed

from .feat_training_methods import (
    build_trw,
    feat_training_alacarte,
    feat_training_doc2vec,
    feat_training_fasttext,
    feat_training_flash,
    feat_training_provd,
    feat_training_trw,
    feat_training_word2vec,
)
from .utils import build_random_walks


def main(cfg):
    set_seed(cfg)

    method = cfg.featurization.feat_training.used_method.strip()
    if method == "alacarte":
        build_random_walks.main(cfg)
        feat_training_alacarte.main(cfg)
    elif method == "doc2vec":
        feat_training_doc2vec.main(cfg)
    elif method in ["hierarchical_hashing", "only_type", "magic", "only_ones"]:
        # these methods don't need to build or train any model
        # so we do nothing here and generate vectorized graphs directly in feat_inference.py
        pass
    elif method == "word2vec":
        feat_training_word2vec.main(cfg)
    elif method == "temporal_rw":
        build_trw.main(cfg)
        feat_training_trw.main(cfg)
    elif method == "flash":
        feat_training_flash.main(cfg)
    elif method == "fasttext":
        feat_training_fasttext.main(cfg)
    elif method == "provd":
        feat_training_provd.main(cfg)
    else:
        raise ValueError(f"Invalid node embedding method {method}")
