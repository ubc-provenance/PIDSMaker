"""DeepWalk / Node2Vec pretraining branch for SPIDER."""

from pidsmaker.utils.utils import log
from ..models.deepwalk import walks_to_label_sentences, train_deepwalk, save_deepwalk


def pretrain_deepwalk(cfg, spider_cfg, walk_corpus, out_dir, model_type):
    emb_dim = cfg.featurization.emb_dim
    seed = cfg.featurization.seed
    dw_cfg = spider_cfg.deepwalk
    dw_window = dw_cfg.window
    dw_epochs = dw_cfg.epochs
    dw_min_count = dw_cfg.min_count
    dw_workers = dw_cfg.workers

    label = "Node2Vec" if model_type == "node2vec" else "DeepWalk"
    if model_type == "node2vec":
        log(f"Node2Vec walks sampled with p={spider_cfg.node2vec.p}, q={spider_cfg.node2vec.q}")

    log(f"Converting walks to label sentences for {label}...")
    sentences = walks_to_label_sentences(walk_corpus)
    log(f"{label} sentences: {len(sentences):,} (from {len(walk_corpus):,} walks)")

    dw_model = train_deepwalk(
        sentences, emb_dim=emb_dim, window=dw_window,
        min_count=dw_min_count, epochs=dw_epochs,
        workers=dw_workers, seed=seed,
    )
    save_deepwalk(dw_model, out_dir)
    log(f"{label} pretraining complete.")
