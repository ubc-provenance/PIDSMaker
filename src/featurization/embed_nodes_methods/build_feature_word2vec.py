import os
from provnet_utils import *
from config import *
from tqdm import tqdm
from gensim.models import Word2Vec
import numpy as np
import random
import torch

def load_corpus_from_database(indexid2msg):

    corpus = {}
    for indexid, msg in tqdm(indexid2msg.items(), desc='Tokenizing corpus from database:'):
        if msg[0] == 'subject':
            tokens = tokenize_subject(msg[1])
        elif msg[0] == 'file':
            tokens = tokenize_file(msg[1])
        else:
            tokens = tokenize_netflow(msg[1])
        corpus[msg[1]] = tokens
    return list(corpus.values())

def train_feature_word2vec(corpus, cfg, model_save_path):
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    window_size = cfg.featurization.embed_nodes.feature_word2vec.window_size
    min_count = cfg.featurization.embed_nodes.feature_word2vec.min_count
    use_skip_gram = cfg.featurization.embed_nodes.feature_word2vec.use_skip_gram
    num_workers = cfg.featurization.embed_nodes.feature_word2vec.num_workers
    epochs = cfg.featurization.embed_nodes.feature_word2vec.epochs
    compute_loss = cfg.featurization.embed_nodes.feature_word2vec.compute_loss
    negative = cfg.featurization.embed_nodes.feature_word2vec.negative
    use_seed = cfg.featurization.embed_nodes.use_seed
    SEED = 0

    if use_seed:
        model = Word2Vec(corpus,
                            vector_size=emb_dim,
                            window=window_size,
                            min_count=min_count,
                            sg=use_skip_gram,
                            workers=num_workers,
                            epochs=1,
                            compute_loss=compute_loss,
                            negative=negative,
                            seed=SEED)
    else:
        model = Word2Vec(corpus,
                            vector_size=emb_dim,
                            window=window_size,
                            min_count=min_count,
                            sg=use_skip_gram,
                            workers=num_workers,
                            epochs=1,
                            compute_loss=compute_loss,
                            negative=negative)
        epoch_loss = model.get_latest_training_loss()
        log(f"Epoch: 0/{epochs}; loss: {epoch_loss}")

        for epoch in range(epochs - 1):
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
    model_save_path = cfg.featurization.embed_nodes.feature_word2vec._model_dir
    os.makedirs(model_save_path, exist_ok=True)

    log(f"Building feature word2vec model and save model to {model_save_path}")

    log(f"Get indexid2msg from database...")
    indexid2msg = get_indexid2msg(cfg)

    log(f"Start building and training feature word2vec model...")

    log("Loading and tokenizing corpus from database...")
    corpus = load_corpus_from_database(indexid2msg=indexid2msg)

    train_feature_word2vec(corpus=corpus,
                           cfg=cfg,
                           model_save_path=model_save_path)

if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)