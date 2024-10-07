from config import *
from .transformation_methods import (
    transformation_rcaid_pseudo_graph,
)

def main(cfg):
    method  = cfg.preprocessing.transformation.used_method
    if method == "none":
        pass
    elif method == "rcaid_pseudo_graph":
        transformation_rcaid_pseudo_graph.main(cfg)
    else:
        raise ValueError(f"Unrecognized transformation method: {method}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
