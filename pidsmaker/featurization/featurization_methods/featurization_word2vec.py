"""Word2Vec-based node featurization.

Learns distributed representations of node labels using Word2Vec on random walks
through the provenance graph. Supports skip-gram and CBOW architectures for
capturing structural and semantic relationships between nodes.
"""

import os

from gensim.models import Word2Vec

from pidsmaker.featurization.featurization_utils import get_corpus
from pidsmaker.utils.utils import log, log_start


def train_word2vec(corpus, cfg, model_save_path):
    """Train Word2Vec model on graph corpus (random walks).

    Args:
        corpus: List of walks (sequences of node labels)
        cfg: Configuration with Word2Vec hyperparameters
        model_save_path: Path to save trained model

    Returns:
        Word2Vec: Trained model
    """
    emb_dim = cfg.featurization.emb_dim
    alpha = cfg.featurization.word2vec.alpha
    window_size = cfg.featurization.word2vec.window_size
    min_count = cfg.featurization.word2vec.min_count
    use_skip_gram = cfg.featurization.word2vec.use_skip_gram
    num_workers = cfg.featurization.word2vec.num_workers
    epochs = cfg.featurization.epochs
    compute_loss = cfg.featurization.word2vec.compute_loss
    negative = cfg.featurization.word2vec.negative
    use_seed = cfg.featurization.use_seed
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
    model_save_path = cfg.featurization._model_dir
    os.makedirs(model_save_path, exist_ok=True)

    log("Loading and tokenizing corpus from database...")
    multi_dataset_training = cfg.featurization.multi_dataset_training

    corpus = get_corpus(cfg, gather_multi_dataset=multi_dataset_training)

    log(f"Building feature word2vec model and save model to {model_save_path}")
    train_word2vec(corpus=corpus, cfg=cfg, model_save_path=model_save_path)
