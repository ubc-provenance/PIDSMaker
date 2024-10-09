from config import *
from provnet_utils import *


def main(cfg, is_test_set: bool=False):
    log_start(__file__)
    base_dir = cfg.preprocessing.transformation._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)
    
    trained_w2v_dir = cfg.featurization.embed_nodes.word2vec._vec_graphs_dir
    if is_test_set:
        indexid2vec = torch.load(os.path.join(trained_w2v_dir, f"nodelabel2vec_test-{file}"))
    else:
        indexid2vec = torch.load(os.path.join(trained_w2v_dir, "nodelabel2vec_val"))  # From both train and val
        
    # TODO: check it works
    return indexid2vec
