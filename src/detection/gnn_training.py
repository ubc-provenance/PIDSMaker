from config import *
from .training_methods import (
    orthrus_gnn_training,
    flash_training,
    threatrace_training,
    magic_testing,
)

def main(cfg):
    method = cfg.detection.gnn_training.used_method.strip()
    if method == 'orthrus':
        orthrus_gnn_training.main(cfg)
    elif method == 'magic':
        magic_testing.main(cfg)
    elif method == 'flash':
        flash_training.main(cfg)
    elif method == "threatrace":
        threatrace_training.main(cfg)
    else:
        raise ValueError(f"Invalid training method {method}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
