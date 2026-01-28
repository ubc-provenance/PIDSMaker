from pidsmaker.featurization.featurization_methods import (
    build_trw,
    featurization_alacarte,
    featurization_doc2vec,
    featurization_fasttext,
    featurization_flash,
    featurization_trw,
    featurization_word2vec,
)
from pidsmaker.featurization.utils import build_random_walks
from pidsmaker.utils.utils import set_seed


def main(cfg):
    set_seed(cfg)

    method = cfg.featurization.used_method.strip()
    if method == "alacarte":
        build_random_walks.main(cfg)
        featurization_alacarte.main(cfg)
    elif method == "doc2vec":
        featurization_doc2vec.main(cfg)
    elif method in ["hierarchical_hashing", "only_type", "magic", "only_ones"]:
        # these methods don't need to build or train any model
        # so we do nothing here and generate vectorized graphs directly in feat_inference.py
        pass
    elif method == "word2vec":
        featurization_word2vec.main(cfg)
    elif method == "temporal_rw":
        build_trw.main(cfg)
        featurization_trw.main(cfg)
    elif method == "flash":
        featurization_flash.main(cfg)
    elif method == "fasttext":
        featurization_fasttext.main(cfg)
    else:
        raise ValueError(f"Invalid node embedding method {method}")
