from provnet_utils import *
from config import *
from featurization.embed_nodes_methods.embed_nodes_flash import get_node2corpus

from gensim.models import Word2Vec


def infer(document, w2vmodel, encoder):
    """
    Each node is associated to a `document` which is the list of (msg => edge type => msg)
    involving this node. 
    We get the embedding of each word inside this document and we do the mean of all embeddings.
    OOV words are simply ignored.
    """
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]

    embedding_dim = w2vmodel.vector_size

    if not word_embeddings:
        return np.zeros(embedding_dim)

    word_embeddings_array = np.array(word_embeddings)

    output_embedding = torch.tensor(word_embeddings_array, dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)

class PositionalEncoder:
    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]

def main(cfg):
    log_start(__file__)
    
    trained_w2v_dir = cfg.featurization.embed_nodes._model_dir
    w2vmodel = Word2Vec.load(os.path.join(trained_w2v_dir, "word2vec_model_final.model"))
    w2v_vector_size = cfg.featurization.embed_nodes.emb_dim

    node2corpus = get_node2corpus(cfg, splits=["train", "val", "test"])
    indexid2vec = {}
    for indexid, corpus in log_tqdm(node2corpus.items(), desc='Embeding all nodes in the dataset'):
        indexid2vec[indexid] = infer(corpus, w2vmodel, PositionalEncoder(w2v_vector_size))

    return indexid2vec
