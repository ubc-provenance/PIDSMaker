from config import *
from . import build_random_walks
from .embed_nodes_methods import (
    embed_nodes_word2vec,
    embed_nodes_doc2vec,
    build_feature_word2vec,
)


def main(cfg):
    method = cfg.featurization.embed_nodes.used_method.strip()
    if method == "word2vec":
        build_random_walks.main(cfg)
        embed_nodes_word2vec.main(cfg)
    elif method == "doc2vec":
        embed_nodes_doc2vec.main(cfg)
    elif method == "feature_word2vec":
        build_feature_word2vec.main(cfg)
    else:
        raise ValueError(f"Invalid node embedding method {method}")


if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
