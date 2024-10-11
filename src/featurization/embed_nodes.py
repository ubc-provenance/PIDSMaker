from config import *
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
    method = cfg.featurization.embed_nodes.used_method.strip()
    if method == "word2vec":
        build_random_walks.main(cfg)
        embed_nodes_word2vec.main(cfg)
    elif method == "doc2vec":
        embed_nodes_doc2vec.main(cfg)
    elif method == "hierarchical_hashing" or method == "only_type" or method == "magic":
        # hierarchical feature hashing doesn't need to build or train any model
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


if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
