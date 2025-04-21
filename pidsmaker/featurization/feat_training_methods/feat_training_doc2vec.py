import os

from gensim.models.doc2vec import Doc2Vec

from pidsmaker.featurization.featurization_utils import (
    get_corpus,
    get_corpus_using_neighbors_features,
)
from pidsmaker.utils.utils import log, log_start


def doc2vec(
    cfg,
    tagged_data: list[str],
    model_save_path: str,
    epochs: int,
    emb_dim: int,
    alpha: float,
    dm: int = 1,
):
    SEED = 0
    model = Doc2Vec(
        vector_size=emb_dim, alpha=alpha, min_count=1, dm=dm, compute_loss=True, seed=SEED
    )
    model.build_vocab(tagged_data)

    log("Training Doc2vec...")
    model.train(tagged_data, total_examples=model.corpus_count, epochs=epochs, compute_loss=True)

    log(f"Saving Doc2Vec model to {model_save_path}")
    os.makedirs(model_save_path, exist_ok=True)
    model.save(os.path.join(model_save_path, "doc2vec_model.model"))
    return model


def main(cfg):
    log_start(__file__)
    model_save_path = cfg.featurization.feat_training._model_dir

    # Context-aware Doc2vec embedding that considers the neighbors when creating embedding (like in Rcaid)
    if cfg.featurization.feat_training.doc2vec.include_neighbors:
        tagged_data = get_corpus_using_neighbors_features(cfg, doc2vec_format=True)

    # Standard token-level Doc2vec
    else:
        tagged_data = get_corpus(cfg, doc2vec_format=True)

    epochs = cfg.featurization.feat_training.epochs
    emb_dim = cfg.featurization.feat_training.emb_dim
    alpha = cfg.featurization.feat_training.doc2vec.alpha

    log("Start building and training Doc2Vec model...")
    doc2vec(
        tagged_data=tagged_data,
        model_save_path=model_save_path,
        epochs=epochs,
        emb_dim=emb_dim,
        alpha=alpha,
        cfg=cfg,
    )
