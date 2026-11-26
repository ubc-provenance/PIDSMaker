"""Doc2Vec-based node featurization with neighborhood awareness.

Doc2Vec learns representations considering node neighborhoods, capturing both node
identity and local graph structure. Used by systems like Orthrus and R-Caid for
context-aware embeddings in provenance graphs.
"""

import os

from gensim.models.doc2vec import Doc2Vec

from pidsmaker.featurization.featurization_utils import (
    get_corpus,
    get_corpus_using_neighbors_features,
)
from pidsmaker.utils.process_tags import truncate_process_name
from pidsmaker.utils.utils import log, log_start


def doc2vec_label_fn(cfg):
    """Function replacing the label of a node before computing its features, or None."""
    if not getattr(cfg.featurization.doc2vec, "truncate_process_name", None):
        return None

    def label_fn(node_type, label):
        return truncate_process_name(label) if node_type == "subject" else label

    return label_fn


def doc2vec(
    cfg,
    tagged_data,
    model_save_path: str,
    epochs: int,
    emb_dim: int,
    alpha: float,
    dm: int = 1,
    workers: int = 3,
):
    """Train Doc2Vec model on graph corpus.

    Args:
        cfg: Configuration object
        tagged_data: Tagged documents (walks with node identifiers), a list or a re-iterable
            corpus (iterated once to build the vocabulary, then once per epoch)
        model_save_path: Path to save model
        epochs: Training epochs
        emb_dim: Embedding dimension
        alpha: Learning rate
        dm: 1 for PV-DM, 0 for PV-DBOW

    Returns:
        Doc2Vec: Trained model
    """
    SEED = 0
    model = Doc2Vec(
        vector_size=emb_dim, alpha=alpha, min_count=1, dm=dm, compute_loss=True, seed=SEED,
        workers=workers,
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
    model_save_path = cfg.featurization._model_dir

    # Context-aware Doc2vec embedding that considers the neighbors when creating embedding (like in Rcaid)
    if cfg.featurization.doc2vec.include_neighbors:
        if doc2vec_label_fn(cfg) is not None:
            raise ValueError("`truncate_process_name` is not supported with `include_neighbors`")
        tagged_data = get_corpus_using_neighbors_features(cfg, doc2vec_format=True)

    # Standard token-level Doc2vec
    else:
        tagged_data = get_corpus(cfg, doc2vec_format=True, label_fn=doc2vec_label_fn(cfg))

    epochs = cfg.featurization.epochs
    emb_dim = cfg.featurization.emb_dim
    alpha = cfg.featurization.doc2vec.alpha

    log("Start building and training Doc2Vec model...")
    doc2vec(
        tagged_data=tagged_data,
        model_save_path=model_save_path,
        epochs=epochs,
        emb_dim=emb_dim,
        alpha=alpha,
        workers=cfg.featurization.doc2vec.workers,
        cfg=cfg,
    )
