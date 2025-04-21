import os

from gensim.models import Word2Vec

from pidsmaker.provnet_utils import (
    get_all_files_from_folders,
    get_indexid2msg,
    log,
    log_start,
    log_tqdm,
    tokenize_file,
    tokenize_netflow,
    tokenize_subject,
)


def tokenize_corpus(corpus, indexid2msg):
    tokenized_corpus = []
    for line in corpus:
        line = line.strip()
        tokenized_line = []
        for node in line.split(","):
            msg = indexid2msg[node]
            if msg[0] == "subject":
                tokens = tokenize_subject(msg[1])
            elif msg[0] == "file":
                tokens = tokenize_file(msg[1])
            else:
                tokens = tokenize_netflow(msg[1])
            tokenized_line.extend(tokens)
        tokenized_corpus.append(tokenized_line)
    return tokenized_corpus


def train_word2vec(corpus, model_save_path, cfg):
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    window_size = cfg.featurization.embed_nodes.temporal_rw.window_size
    min_count = cfg.featurization.embed_nodes.temporal_rw.min_count
    use_skip_gram = cfg.featurization.embed_nodes.temporal_rw.use_skip_gram
    num_workers = cfg.featurization.embed_nodes.temporal_rw.wv_workers
    epochs = cfg.featurization.embed_nodes.epochs
    compute_loss = cfg.featurization.embed_nodes.temporal_rw.compute_loss
    negative = cfg.featurization.embed_nodes.temporal_rw.negative
    use_seed = cfg.featurization.embed_nodes.use_seed
    SEED = 0

    model = Word2Vec(
        corpus,
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
        model.train(corpus, epochs=1, total_examples=len(corpus), compute_loss=compute_loss)
        epoch_loss = model.get_latest_training_loss()
        log(f"Epoch: {epoch + 1}/{epochs}; loss: {epoch_loss}")

    loss = model.get_latest_training_loss()
    log(f"Epoch: {epochs}; loss: {loss}")

    model.init_sims(replace=True)
    model.save(os.path.join(model_save_path, "trw_word2vec.model"))
    log(f"Save word2vec to {os.path.join(model_save_path, 'trw_word2vec.model')}")


def update_word2vec(corpus, model_save_path, cfg):
    epochs = cfg.featurization.embed_nodes.epochs
    compute_loss = cfg.featurization.embed_nodes.temporal_rw.compute_loss

    log(f"Loading word2vec from {model_save_path}")
    model = Word2Vec.load(os.path.join(model_save_path, "trw_word2vec.model"))
    model.build_vocab(corpus, update=True)

    for epoch in range(epochs):
        model.train(corpus, epochs=1, total_examples=len(corpus), compute_loss=compute_loss)
        epoch_loss = model.get_latest_training_loss()
        log(f"Epoch: {epoch}/{epochs}; loss: {epoch_loss}")

    model.init_sims(replace=True)
    model.save(os.path.join(model_save_path, "trw_word2vec.model"))
    log(f"Save word2vec to {os.path.join(model_save_path, 'trw_word2vec.model')}")


def main(cfg):
    log_start(__file__)
    model_save_path = cfg.featurization.embed_nodes._model_dir
    os.makedirs(model_save_path, exist_ok=True)

    log(f"Building TRW based word2vec model and save model to {model_save_path}")

    log("Get indexid2msg from database...")
    indexid2msg = get_indexid2msg(cfg)

    corpus_base_dir = cfg.featurization.embed_nodes.temporal_rw._random_walk_corpus_dir
    corpus_folders = ["train", "val", "test", "unused"]
    corpus_file_list = get_all_files_from_folders(corpus_base_dir, corpus_folders)
    for i in log_tqdm(
        list(range(len(corpus_file_list))), desc="Train Word2Vec based on TRW corpus:"
    ):
        with open(corpus_file_list[i], "r") as f:
            corpus = f.readlines()
            log(f"\n{len(corpus)} lines read from {corpus_file_list[i]}")

            tokenized_corpus = tokenize_corpus(corpus, indexid2msg)

            if i == 0:
                train_word2vec(corpus=tokenized_corpus, model_save_path=model_save_path, cfg=cfg)
            else:
                update_word2vec(corpus=tokenized_corpus, model_save_path=model_save_path, cfg=cfg)
