import os

import wget
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model

from pidsmaker.config import ROOT_ARTIFACT_DIR
from pidsmaker.featurization.featurization_utils import get_corpus
from pidsmaker.utils.utils import log, log_start


def download_facebook_weights(out_dir):
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"

    if not os.path.exists(out_dir):
        wget.download(url, out_dir)


def train_fasttext(corpus, cfg):
    emb_dim = cfg.featurization.feat_training.emb_dim
    epochs = cfg.featurization.feat_training.epochs
    window_size = cfg.featurization.feat_training.fasttext.window_size
    alpha = cfg.featurization.feat_training.fasttext.alpha
    min_count = cfg.featurization.feat_training.fasttext.min_count
    num_workers = cfg.featurization.feat_training.fasttext.num_workers
    negative = cfg.featurization.feat_training.fasttext.negative
    use_seed = cfg.featurization.feat_training.use_seed
    SEED = 0

    use_pretrained_fb_model = cfg.featurization.feat_training.fasttext.use_pretrained_fb_model

    if use_pretrained_fb_model:
        out_dir = os.path.join(ROOT_ARTIFACT_DIR, "fasttext_facebook_cc.en.300.bin.gz")

        log("Downloading Facebook's FastText model...")
        download_facebook_weights(out_dir)
        log("Loading Facebook's FastText model...")
        model = load_facebook_model(out_dir)

    else:
        model = FastText(
            min_count=min_count,
            vector_size=emb_dim,
            workers=num_workers,
            alpha=alpha,
            window=window_size,
            negative=negative,
            seed=SEED,
        )

    model.build_vocab(corpus, update=use_pretrained_fb_model)
    model.train(corpus, epochs=epochs, total_examples=len(corpus))

    return model


def main(cfg):
    log_start(__file__)

    multi_dataset_training = cfg.featurization.feat_training.multi_dataset_training
    corpus = get_corpus(cfg, gather_multi_dataset=multi_dataset_training)

    log("Training FastText model...")
    model = train_fasttext(
        corpus=corpus,
        cfg=cfg,
    )

    model_save_path = cfg.featurization.feat_training._model_dir
    os.makedirs(model_save_path, exist_ok=True)
    model.save(os.path.join(model_save_path, "fasttext.pkl"))
