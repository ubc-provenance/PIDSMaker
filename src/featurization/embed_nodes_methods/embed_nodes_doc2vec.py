from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from nltk.tokenize import word_tokenize

from provnet_utils import *

def splitting_label_set(split_files: list[str], cfg):
    base_dir = cfg.preprocessing.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)

    node_set = set()

    for path in sorted_paths:
        graph = torch.load(path)
        nodes = graph.nodes()
        node_set = node_set | set(nodes)

    return list(node_set)  #list[str]

def preprocess(indexid2msg: dict, nodes: list[str]):

    tags = []
    words = []

    for node in nodes:
        node_type, msg = indexid2msg[int(node)]
        tags.append(node)

        if node_type == 'subject':
            words.append(tokenize_subject(msg))
        if node_type == 'file':
            words.append(tokenize_file(msg))
        if node_type == 'netflow':
            words.append(tokenize_netflow(msg))

    return words, tags


def doc2vec(train_set: list[str],
                  model_save_path: str,
                  indexid2msg: dict,
                  logger: logging.Logger,
                  epochs: int,
                  emb_dim: int,
                  alpha: float,
                  min_alpha: float,
                  dm: int = 1):

    logger.info('Preprocessing training data...')
    words, tags = preprocess(indexid2msg, train_set)
    tagged_data = [TaggedDocument(words=word_list, tags=[tag]) for word_list, tag in zip(words, tags)]

    logger.info('Initializing Doc2Vec model...')
    model = Doc2Vec(vector_size=emb_dim, alpha=alpha, min_count=1, dm=dm, compute_loss=True)
    model.build_vocab(tagged_data)

    logger.info('Start training...')

    for epoch in range(epochs):
        model.train(tagged_data, total_examples=len(words), epochs=1, compute_loss=True)
        model.alpha -= 0.0002
        if model.alpha < min_alpha:
            model.alpha = min_alpha
        logger.info(f'Epoch {epoch} / {epochs}, Training loss: {model.get_latest_training_loss()}')
        print(f'Epoch {epoch} / {epochs}, Training loss: {model.get_latest_training_loss()}')

    logger.info(f'Saving Doc2Vec model to {model_save_path}')
    print(f'Saving Doc2Vec model to {model_save_path}')
    model.save(model_save_path + 'doc2vec_model.model')
    pass

def main(cfg):
    model_save_path = cfg.featurization.embed_nodes.doc2vec._model_dir
    os.makedirs(model_save_path,exist_ok=True)

    logger = get_logger(
        name="doc2vec",
        filename=os.path.join(cfg.featurization.embed_nodes.doc2vec._logs_dir, "doc2vec.log")
    )
    logger.info(f"Building doc2vec and save model to {model_save_path}")

    logger.info(f"Get indexid2msg from database...")
    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur)

    logger.info(f"Splitting datasets...")
    train_set_nodes = splitting_label_set(split_files=cfg.dataset.train_files, cfg=cfg)
    # val_set_nodes = splitting_label_set(split_files=cfg.dataset.val_files, cfg=cfg)
    # test_set_nodes = splitting_label_set(split_files=cfg.dataset.test_files, cfg=cfg)

    epochs = cfg.featurization.embed_nodes.doc2vec.epochs
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    alpha = cfg.featurization.embed_nodes.doc2vec.alpha
    min_alpha = cfg.featurization.embed_nodes.doc2vec.min_alpha

    logger.info(f"Start building and training Doc2Vec model...")
    print(f"Start building and training Doc2Vec model...")
    doc2vec(train_set=train_set_nodes,
                  model_save_path=model_save_path,
                  indexid2msg=indexid2msg,
                  logger=logger,
                  epochs=epochs,
                  emb_dim=emb_dim,
                  alpha=alpha,
                  min_alpha=min_alpha)

if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
