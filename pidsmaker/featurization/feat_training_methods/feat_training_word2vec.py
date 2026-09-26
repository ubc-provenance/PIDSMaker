import os

from gensim.models import Word2Vec

from pidsmaker.featurization.featurization_utils import get_corpus, get_corpus_from_pretrain_datasets
from pidsmaker.utils.utils import log, log_start


def train_word2vec(corpus, cfg, model_save_path):
    if cfg._from_weights:
        path = os.path.join(cfg._from_weights_path, "word2vec.model")
        model = Word2Vec.load(path)
    
    else:
        emb_dim = cfg.featurization.feat_training.emb_dim
        window_size = cfg.featurization.feat_training.word2vec.window_size
        min_count = cfg.featurization.feat_training.word2vec.min_count
        use_skip_gram = cfg.featurization.feat_training.word2vec.use_skip_gram
        num_workers = cfg.featurization.feat_training.word2vec.num_workers
        epochs = cfg.featurization.feat_training.epochs
        compute_loss = cfg.featurization.feat_training.word2vec.compute_loss
        negative = cfg.featurization.feat_training.word2vec.negative
        seed = cfg.featurization.feat_training.seed

        model = Word2Vec(corpus,
                            vector_size=emb_dim,
                            window=window_size,
                            min_count=min_count,
                            sg=use_skip_gram,
                            workers=num_workers,
                            epochs=1,
                            compute_loss=compute_loss,
                            negative=negative,
                            seed=seed)

        epoch_loss = model.get_latest_training_loss()
        log(f"Epoch: 0/{epochs}; loss: {epoch_loss}")

        for epoch in range(epochs - 1):
            model.train(corpus, epochs=1, total_examples=len(corpus), compute_loss=compute_loss)
            epoch_loss = model.get_latest_training_loss()
            log(f"Epoch: {epoch+1}/{epochs}; loss: {epoch_loss}")

        loss = model.get_latest_training_loss()
        log(f"Epoch: {epochs}; loss: {loss}")

        model.init_sims(replace=True)
    
    model.save(os.path.join(model_save_path, 'word2vec.model'), sep_limit=10**10) # force to save in a single file
    log(f"Save word2vec to {os.path.join(model_save_path, 'word2vec.model')}")


def main(cfg):
    log_start(__file__)
    model_save_path = cfg.featurization.feat_training._model_dir
    os.makedirs(model_save_path, exist_ok=True)

    log("Loading and tokenizing corpus from database...")
    pretrain_datasets = cfg.featurization.feat_training.pretrain_datasets
    multi_dataset_training = cfg.featurization.feat_training.multi_dataset_training

    if cfg._from_weights:
        corpus = None
    elif pretrain_datasets:
        corpus = get_corpus_from_pretrain_datasets(cfg)
    else:
        corpus = get_corpus(cfg, gather_multi_dataset=multi_dataset_training)

    log(f"Building feature word2vec model and save model to {model_save_path}")
    train_word2vec(corpus=corpus, cfg=cfg, model_save_path=model_save_path)
