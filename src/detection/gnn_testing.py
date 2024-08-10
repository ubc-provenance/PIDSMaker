from config import *
from .testing_methods import (
    orthrus_gnn_testing,
    flash_testing,
)

def main(cfg):
    method = cfg.detection.gnn_training.used_method.strip()
    if method == 'orthrus':
        orthrus_gnn_testing.main(cfg)
    elif method == 'magic':
        set_task_to_done(cfg.detection.gnn_testing._task_path)
    elif method == 'flash':
        flash_testing.main(cfg)
    else:
        raise ValueError(f"Invalid testing method {method}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
