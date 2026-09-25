from pidsmaker.detection.training_methods import (
    training_loop,
)
from pidsmaker.utils.utils import log_start


def main(cfg):
    log_start(__file__)
    method = cfg.training.used_method.strip()
    if method == "default":
        return training_loop.main(cfg)
    elif method == "spider":
        from pidsmaker.spider import detect
        return detect.main(cfg)
    else:
        raise ValueError(f"Invalid training method {method}")
