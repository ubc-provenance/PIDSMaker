from pidsmaker.detection.training_methods import (
    training_loop,
)


def main(cfg):
    method = cfg.gnn_training.used_method.strip()
    if method == "default":
        return training_loop.main(cfg)
    else:
        raise ValueError(f"Invalid training method {method}")
