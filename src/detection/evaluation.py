from .evaluation_methods import (
    node_evaluation,
    queue_evaluation,
)


def main(cfg):
    if "node_evaluation" in cfg.detection.evaluation.used_method:
        node_evaluation.main(cfg)
    elif "queue_evaluation" in cfg.detection.evaluation.used_method:
        queue_evaluation.main(cfg)
    else:
        raise ValueError(f"Invalid evaluation method {cfg.detection.evaluation.used_method}")
        

if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
