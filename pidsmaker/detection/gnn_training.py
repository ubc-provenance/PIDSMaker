from pidsmaker.detection.training_methods import (
    provd_testing,
    training_loop,
)


def main(cfg):
    method = cfg.detection.gnn_training.used_method.strip()
    if method == "default":
        return training_loop.main(cfg)
    elif method == "provd":
        return provd_testing.main(cfg)
    else:
        raise ValueError(f"Invalid training method {method}")
