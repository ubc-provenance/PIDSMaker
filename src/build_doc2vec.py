from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from nltk.tokenize import word_tokenize

from provnet_utils import *

def get_indexid2msg(cur):
    indexid2msg = {}

    # netflow
    sql = """
        select * from netflow_node_table;
        """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        remote_address = i[4] + ':' + i[5]
        index_id = i[-1] # int
        indexid2msg[index_id] = ['netflow', remote_address]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        path = i[2]
        cmd = i[3]
        index_id = i[-1]
        indexid2msg[index_id] = ['subject', path + ' ' +cmd]

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        path = i[2]
        index_id = i[-1]
        indexid2msg[index_id] = ['file', path]

    return indexid2msg #{index_id: [node_type, msg]}

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
    def tokenize_subject(sentence: str):
        return word_tokenize(sentence.replace('/',' ').replace('=',' = ').replace(':',' : '))
    def tokenize_file(sentence: str):
        return word_tokenize(sentence.replace('/',' '))
    def tokenize_netflow(sentence: str):
        return word_tokenize(sentence.replace(':',' ').replace('.',' '))

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


def build_doc2vec(train_set: list[str],
                  model_save_path: str,
                  indexid2msg: dict,
                  logger: logging.Logger,
                  epochs: int,
                  vec_size: int,
                  alpha: float,
                  min_alpha: float,
                  dm: int = 1):

    logger.info('Preprocessing training data...')
    words, tags = preprocess(indexid2msg, train_set)
    tagged_data = [TaggedDocument(words=word_list, tags=[tag]) for word_list, tag in zip(words, tags)]

    logger.info('Initializing Doc2Vec model...')
    model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_count=1, dm=dm, compute_loss=True)
    model.build_vocab(tagged_data)

    logger.info('Start training...')

    for epoch in range(epochs):
        model.train(tagged_data, total_examples=len(words), epochs=1, compute_loss=True)
        model.alpha -= 0.0002
        if model.alpha < min_alpha:
            model.alpha = min_alpha
        logger.info(f'Epoch {epoch} / {epochs}, Training loss: {model.get_latest_training_loss()}')

    logger.info(f'Saving Doc2Vec model to {model_save_path}')
    model.save(model_save_path + 'doc2vec_model.model')
    pass

def main(cfg):
    #TODO: modify model saving dir
    model_save_path = os.path.join(cfg._artifact_dir,"doc2vec/")
    os.makedirs(model_save_path,exist_ok=True)

    #TODO: modify logger dir
    logger = get_logger(
        name="build_doc2vec",
        filename=os.path.join(cfg._artifact_dir, "doc2vec.log")
    )
    logger.info(f"Building doc2vec and save model to {model_save_path}")

    logger.info(f"Get indexid2msg from database...")
    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur)

    logger.info(f"Splitting datasets...")
    train_set_nodes = splitting_label_set(split_files=cfg.dataset.train_files, cfg=cfg)
    # val_set_nodes = splitting_label_set(split_files=cfg.dataset.val_files, cfg=cfg)
    # test_set_nodes = splitting_label_set(split_files=cfg.dataset.test_files, cfg=cfg)

    #TODO: move parameters to config file
    epochs = 100
    vec_size = 128
    alpha = 0.025
    min_alpha = 0.00025

    logger.info(f"Start building and training Doc2Vec model...")
    build_doc2vec(train_set=train_set_nodes,
                  model_save_path=model_save_path,
                  indexid2msg=indexid2msg,
                  logger=logger,
                  epochs=epochs,
                  vec_size=vec_size,
                  alpha=alpha,
                  min_alpha=min_alpha)

if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
