from config import *
from provnet_utils import *


def main(cfg, is_test_set: bool=False):
    log_start(__file__)
    
    # TODO: this part is broken, need some work to make it work (the training part of this method seems too complicated)
    
    trained_w2v_dir = cfg.featurization.embed_nodes.word2vec._vec_graphs_dir
    if is_test_set:
        indexid2vec = torch.load(os.path.join(trained_w2v_dir, f"nodelabel2vec_test-{file}"))
    else:
        indexid2vec = torch.load(os.path.join(trained_w2v_dir, "nodelabel2vec_val"))  # From both train and val
        
    return indexid2vec
