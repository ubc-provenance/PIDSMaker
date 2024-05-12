from config import *

from .embed_edges_methods import (
    embed_edges_word2vec,
    embed_edges_doc2vec,
)


def main(cfg):
    method = cfg.featurization.embed_nodes.used_method.strip()
    if method == "word2vec":
        embed_edges_word2vec.main(cfg)
    elif method == "doc2vec":
        embed_edges_doc2vec.main(cfg)
    else:
        raise ValueError(f"Invalid node embedding method {method}")


if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)