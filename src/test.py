from provnet_utils import *
from config import *

from detection.evaluation_methods.evaluation_utils import compute_tw_labels_for_magic


def main(cfg):
    out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
    print(out_dir)





if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)