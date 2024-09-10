from provnet_utils import *
from config import *

from flash_utils.flash_featurization import EpochLogger, EpochSaver
from flash_utils.utils import load_graph_data
from gensim.models import Word2Vec
import os

def get_corpus(cfg):
    corpus = []

    data = load_graph_data(t='train', cfg=cfg)
    for d in data:
        corpus.append(d[0])
    return corpus

class RepeatableIterator:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for phrases in self.data:
            for sentence in phrases:
                yield sentence

def main(cfg):
    log_start(__file__)
    model_save_dir = cfg.featurization.embed_nodes.flash._model_dir
    os.makedirs(model_save_dir, exist_ok=True)

    # logger = EpochLogger()
    # saver = EpochSaver(model_save_dir)

    all_phrases = get_corpus(cfg=cfg)

    # Get hyper args
    vector_size = cfg.featurization.embed_nodes.flash.vector_size
    window = cfg.featurization.embed_nodes.flash.window
    min_count = cfg.featurization.embed_nodes.flash.min_count
    workers = cfg.featurization.embed_nodes.flash.workers
    epochs = cfg.featurization.embed_nodes.flash.epochs

    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)

    model.build_vocab(RepeatableIterator(all_phrases), progress_per=10000)

    total_examples = model.corpus_count

    for epoch in range(epochs):
        log(f"Epoch #{epoch} start")
        model.train(RepeatableIterator(all_phrases), total_examples=total_examples, epochs=1)
        log(f"Epoch #{epoch} end")
        model.save(os.path.join(model_save_dir, f"word2vec_model.model"))
        log(f"Epoch #{epoch} model saved")

    model.save(os.path.join(model_save_dir, "word2vec_model_final.model"))


if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)