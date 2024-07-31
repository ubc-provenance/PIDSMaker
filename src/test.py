from provnet_utils import *
from config import *

def main(cfg):

    print(cfg.preprocessing.build_graphs.used_method)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)