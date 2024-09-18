from config import *

from .embed_edges_methods import (
    embed_edges_word2vec,
    embed_edges_doc2vec,
    embed_edges_HFH,
    embed_edges_feature_word2vec,
    embed_edges_only_type,
    embed_edges_TRW,
)


def main(cfg):
    method = cfg.featurization.embed_nodes.used_method.strip()
    if method == "word2vec":
        embed_edges_word2vec.main(cfg)
    elif method == "doc2vec":
        embed_edges_doc2vec.main(cfg)
    elif method == "hierarchical_hashing":
        embed_edges_HFH.main(cfg)
    elif method == "feature_word2vec":
        embed_edges_feature_word2vec.main(cfg)
    elif method == "only_type":
        embed_edges_only_type.main(cfg)
    elif method == "temporal_rw":
        embed_edges_TRW.main(cfg)
    elif method == "flash" or method == "threatrace" or method == 'magic':
        set_task_to_done(cfg.featurization.embed_edges._task_path)
    else:
        raise ValueError(f"Invalid node embedding method {method}")


if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)