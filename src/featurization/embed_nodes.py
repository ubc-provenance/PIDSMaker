from config import *
from provnet_utils import set_seed
from . import build_random_walks
from .embed_nodes_methods import (
    embed_nodes_word2vec,
    embed_nodes_doc2vec,
    build_feature_word2vec,
    build_temporal_random_walk,
    embed_nodes_trw,
    embed_nodes_flash,
    embed_nodes_fasttext,
    embed_paths_provd,
)


def main(cfg):
    set_seed(cfg)
    
    method = cfg.featurization.embed_nodes.used_method.strip()
    if method == "word2vec":
        build_random_walks.main(cfg)
        embed_nodes_word2vec.main(cfg)
    elif method == "doc2vec":
        embed_nodes_doc2vec.main(cfg)
    elif method in ["hierarchical_hashing", "only_type", "magic", "only_ones"]:
        # these methods don't need to build or train any model
        # so we do nothing here and generate vectorized graphs directly in embed_edges.py
        pass
    elif method == "feature_word2vec":
        build_feature_word2vec.main(cfg)
    elif method == "temporal_rw":
        build_temporal_random_walk.main(cfg)
        embed_nodes_trw.main(cfg)
    elif method == "flash":
        embed_nodes_flash.main(cfg)
    elif method == "fasttext":
        embed_nodes_fasttext.main(cfg)
    elif method == "provd":
        embed_paths_provd.main(cfg)
    else:
        raise ValueError(f"Invalid node embedding method {method}")
