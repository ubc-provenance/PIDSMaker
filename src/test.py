from labelling import get_GP_of_each_attack
from config import *
from provnet_utils import *

if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    get_GP_of_each_attack(cfg)