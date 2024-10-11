from provnet_utils import *
from config import *
from featurization.featurization_utils import get_corpus

from gensim.models import FastText


def train_fasttext(corpus, cfg, model_save_path):
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    epochs = cfg.featurization.embed_nodes.epochs
    window_size = cfg.featurization.embed_nodes.fasttext.window_size
    alpha = cfg.featurization.embed_nodes.fasttext.alpha
    min_count = cfg.featurization.embed_nodes.fasttext.min_count
    num_workers = cfg.featurization.embed_nodes.fasttext.num_workers
    negative = cfg.featurization.embed_nodes.fasttext.negative
    use_seed = cfg.featurization.embed_nodes.use_seed
    SEED = 0

    model = FastText(
        min_count=min_count,
        vector_size=embedding_size,
        workers=num_workers,
        alpha=alpha,
        window=window_size,
        negative=negative,
        seed=SEED,
    )
    model.build_vocab(corpus)
    model.train(corpus, epochs=epochs, total_examples=model.corpus_count)
    
    return model

def main(cfg):
    log_start(__file__)

    indexid2msg = get_indexid2msg(cfg)
    corpus = get_corpus(cfg)

    log("Training FastText model...")
    model = train_fasttext(
        corpus=corpus,
        cfg=cfg,
        model_save_path=model_save_path,
    )
    
    model_save_path = cfg.featurization.embed_nodes._model_dir
    os.makedirs(model_save_path, exist_ok=True)
    model.save(os.path.join(model_save_path, 'fasttext.pkl'))

if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
