from pidsmaker.detection.training_methods import (
    training_loop,
)


def main(cfg):
    method = cfg.training.used_method.strip()
    if method == "default":
        return training_loop.main(cfg)
    elif method == "threatrace":
        from pidsmaker.detection.training_methods import threatrace_training
        return threatrace_training.main(cfg)
    elif method == "pretrained":
        from pidsmaker.pretrained import detect
        return detect.main(cfg)
    else:
        raise ValueError(f"Invalid training method {method}")
