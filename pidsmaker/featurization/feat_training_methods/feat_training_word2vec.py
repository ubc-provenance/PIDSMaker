import os

from gensim.models import Word2Vec

from pidsmaker.featurization.featurization_utils import get_corpus
from pidsmaker.utils.utils import log, log_start


def train_word2vec(corpus, cfg, model_save_path):
    emb_dim = cfg.feat_training.emb_dim
    alpha = cfg.feat_training.word2vec.alpha
    window_size = cfg.feat_training.word2vec.window_size
    min_count = cfg.feat_training.word2vec.min_count
    use_skip_gram = cfg.feat_training.word2vec.use_skip_gram
    num_workers = cfg.feat_training.word2vec.num_workers
    epochs = cfg.feat_training.epochs
    compute_loss = cfg.feat_training.word2vec.compute_loss
    negative = cfg.feat_training.word2vec.negative
    use_seed = cfg.feat_training.use_seed
    SEED = 0

    model = Word2Vec(
        corpus,
        alpha=alpha,
        vector_size=emb_dim,
        window=window_size,
        min_count=min_count,
        sg=use_skip_gram,
        workers=num_workers,
        epochs=1,
        compute_loss=compute_loss,
        negative=negative,
        seed=SEED,
    )

    epoch_loss = model.get_latest_training_loss()
    log(f"Epoch: 0/{epochs}; loss: {epoch_loss}")

    for epoch in range(epochs - 1):
        model.running_training_loss = 0
        model.train(corpus, epochs=1, total_examples=len(corpus), compute_loss=compute_loss)
        epoch_loss = model.get_latest_training_loss()
        log(f"Epoch: {epoch + 1}/{epochs}; loss: {epoch_loss}")

    loss = model.get_latest_training_loss()
    log(f"Epoch: {epochs}; loss: {loss}")

    model.init_sims(replace=True)
    model.save(os.path.join(model_save_path, "word2vec.model"))
    log(f"Save word2vec to {os.path.join(model_save_path, 'word2vec.model')}")


def main(cfg):
    log_start(__file__)
    model_save_path = cfg.feat_training._model_dir
    os.makedirs(model_save_path, exist_ok=True)

    log("Loading and tokenizing corpus from database...")
    multi_dataset_training = cfg.feat_training.multi_dataset_training

    corpus = get_corpus(cfg, gather_multi_dataset=multi_dataset_training)

    log(f"Building feature word2vec model and save model to {model_save_path}")
    train_word2vec(corpus=corpus, cfg=cfg, model_save_path=model_save_path)
