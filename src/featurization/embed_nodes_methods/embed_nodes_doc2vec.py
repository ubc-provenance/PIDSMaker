from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from nltk.tokenize import word_tokenize

from provnet_utils import *
from config import *
import torch
import numpy as np
import random

from featurization.featurization_utils import *


def doc2vec(cfg,
            tagged_data: list[str],
            model_save_path: str,
            indexid2msg: dict,
            epochs: int,
            emb_dim: int,
            alpha: float,
            dm: int = 1,
):
    SEED = 0
    model = Doc2Vec(vector_size=emb_dim, alpha=alpha, min_count=1, dm=dm, compute_loss=True, seed=SEED)
    model.build_vocab(tagged_data)

    log("Training Doc2vec...")
    model.train(tagged_data, total_examples=model.corpus_count, epochs=epochs, compute_loss=True)

    log(f'Saving Doc2Vec model to {model_save_path}')
    model.save(os.path.join(model_save_path, 'doc2vec_model.model'))


def main(cfg):
    log_start(__file__)
    model_save_path = cfg.featurization.embed_nodes.doc2vec._model_dir
    os.makedirs(model_save_path,exist_ok=True)

    indexid2msg = get_indexid2msg(cfg)
    
    # Context-aware Doc2vec embedding that considers the neighbors when creating embedding (like in Rcaid)
    if cfg.featurization.embed_nodes.doc2vec.include_neighbors:
        tagged_data = get_corpus_using_neighbors_features(cfg, doc2vec_format=True)
    
    # Standard token-level Doc2vec
    else:
        tagged_data = get_corpus(cfg, doc2vec_format=True)

    epochs = cfg.featurization.embed_nodes.epochs
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    alpha = cfg.featurization.embed_nodes.doc2vec.alpha

    log(f"Start building and training Doc2Vec model...")
    doc2vec(tagged_data=tagged_data,
                  model_save_path=model_save_path,
                  indexid2msg=indexid2msg,
                  epochs=epochs,
                  emb_dim=emb_dim,
                  alpha=alpha,
                  cfg=cfg)

if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
