from provnet_utils import *
from config import *
from featurization.featurization_utils import get_corpus

import wget
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
from gensim.test.utils import datapath

def download_facebook_weights(out_dir):
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'

    if not os.path.exists(out_dir):
        wget.download(url, out_dir)


def train_fasttext(corpus, cfg):
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    epochs = cfg.featurization.embed_nodes.epochs
    window_size = cfg.featurization.embed_nodes.fasttext.window_size
    alpha = cfg.featurization.embed_nodes.fasttext.alpha
    min_count = cfg.featurization.embed_nodes.fasttext.min_count
    num_workers = cfg.featurization.embed_nodes.fasttext.num_workers
    negative = cfg.featurization.embed_nodes.fasttext.negative
    use_seed = cfg.featurization.embed_nodes.use_seed
    SEED = 0
    
    use_pretrained_fb_model = cfg.featurization.embed_nodes.fasttext.use_pretrained_fb_model
    
    if use_pretrained_fb_model:
        out_dir = os.path.join(ROOT_ARTIFACT_DIR, 'fasttext_facebook_cc.en.300.bin.gz')
        
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

    multi_dataset_training = cfg.featurization.embed_nodes.multi_dataset_training
    corpus = get_corpus(cfg, gather_multi_dataset=multi_dataset_training)

    log("Training FastText model...")
    model = train_fasttext(
        corpus=corpus,
        cfg=cfg,
    )
    
    model_save_path = cfg.featurization.embed_nodes._model_dir
    os.makedirs(model_save_path, exist_ok=True)
    model.save(os.path.join(model_save_path, 'fasttext.pkl'))
