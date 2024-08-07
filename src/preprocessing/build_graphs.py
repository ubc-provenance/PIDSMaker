from config import *
from .build_graph_methods import (
    build_orthrus_graphs,
    build_magic_graphs,
)

def main(cfg):
    graph_method  = cfg.preprocessing.build_graphs.used_method
    if graph_method == "orthrus":
        build_orthrus_graphs.main(cfg)
    elif graph_method == "magic":
        build_magic_graphs.main(cfg)
    else:
        raise ValueError(f"Unrecognized graph method: {graph_method}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
