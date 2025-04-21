from pidsmaker.detection.training_methods import (
    orthrus_gnn_training,
    provd_testing,
)


def main(cfg):
    method = cfg.detection.gnn_training.used_method.strip()
    if method == "orthrus":
        return orthrus_gnn_training.main(cfg)
    elif method == "provd":
        return provd_testing.main(cfg)
    else:
        raise ValueError(f"Invalid training method {method}")
