from provnet_utils import *
from config import *

from labelling import get_ground_truth
from detection.evaluation_methods.evaluation_utils import *

def main(cfg):
    log("Get ground truth")
    GP_nids, _, _ = get_ground_truth(cfg)
    GPs = [str(nid) for nid in GP_nids]
    print(f"There are {len(GPs)} malicious nodes")

    tw_to_malicious_nodes = compute_tw_labels(cfg)
    for tw, mn in tw_to_malicious_nodes.items():
        print(f"tw_to_malicious_nodes: {tw}, {mn}")



if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)