import os
from provnet_utils import *
from config import *
from tqdm import tqdm
from gensim.models import Word2Vec
import numpy as np
import random
import torch

from featurization.featurization_utils import *


def train_feature_word2vec(corpus, cfg, model_save_path):
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    alpha = cfg.featurization.embed_nodes.feature_word2vec.alpha
    window_size = cfg.featurization.embed_nodes.feature_word2vec.window_size
    min_count = cfg.featurization.embed_nodes.feature_word2vec.min_count
    use_skip_gram = cfg.featurization.embed_nodes.feature_word2vec.use_skip_gram
    num_workers = cfg.featurization.embed_nodes.feature_word2vec.num_workers
    epochs = cfg.featurization.embed_nodes.epochs
    compute_loss = cfg.featurization.embed_nodes.feature_word2vec.compute_loss
    negative = cfg.featurization.embed_nodes.feature_word2vec.negative
    use_seed = cfg.featurization.embed_nodes.use_seed
    SEED = 0

    model = Word2Vec(corpus,
                        alpha=alpha,
                        vector_size=emb_dim,
                        window=window_size,
                        min_count=min_count,
                        sg=use_skip_gram,
                        workers=num_workers,
                        epochs=1,
                        compute_loss=compute_loss,
                        negative=negative,
                        seed=SEED)

    epoch_loss = model.get_latest_training_loss()
    log(f"Epoch: 0/{epochs}; loss: {epoch_loss}")

    for epoch in range(epochs - 1):
        model.running_training_loss = 0
        model.train(corpus, epochs=1, total_examples=len(corpus), compute_loss=compute_loss)
        epoch_loss = model.get_latest_training_loss()
        log(f"Epoch: {epoch+1}/{epochs}; loss: {epoch_loss}")

    loss = model.get_latest_training_loss()
    log(f"Epoch: {epochs}; loss: {loss}")

    model.init_sims(replace=True)
    model.save(os.path.join(model_save_path, 'feature_word2vec.model'))
    log(f"Save word2vec to {os.path.join(model_save_path, 'feature_word2vec.model')}")

def main(cfg):
    log_start(__file__)
    model_save_path = cfg.featurization.embed_nodes._model_dir
    os.makedirs(model_save_path, exist_ok=True)

    log("Loading and tokenizing corpus from database...")
    corpus = get_corpus(cfg)

    log(f"Building feature word2vec model and save model to {model_save_path}")
    train_feature_word2vec(corpus=corpus,
                           cfg=cfg,
                           model_save_path=model_save_path)

if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
