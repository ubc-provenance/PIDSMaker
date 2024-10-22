from config import *
from .training_methods import (
    orthrus_gnn_training,
    magic_testing,
    provd_testing,
)

def main(cfg):
    method = cfg.detection.gnn_training.used_method.strip()
    if method == 'orthrus':
        return orthrus_gnn_training.main(cfg)
    elif method == 'magic':
        return magic_testing.main(cfg)
    elif method == "provd":
        return provd_testing.main(cfg)
    else:
        raise ValueError(f"Invalid training method {method}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
