from provnet_utils import *
from config import *

from sklearn.feature_extraction import FeatureHasher


def path2higlist(p):
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
    return l

def ip2higlist(p):
    l=[]
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
    return l

def list2str(l):
    s=''
    for i in l:
        s+=i
    return s

def main(cfg):
    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur)
    
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    FH_string = FeatureHasher(n_features=emb_dim, input_type="string")

    indexid2vec = {}
    for indexid, msg in tqdm(indexid2msg.items(), desc="Embeding all nodes in the dataset"):
        node_type, node_label = msg[0], msg[1]
        if node_type == 'subject' or node_type == 'file':
            higlist = path2higlist(node_label)
        else:
            higlist = ip2higlist(node_label)
        higstr = list2str(higlist)

        dense_vector = FH_string.fit_transform([higstr]).toarray()
        
        normalized_vector = dense_vector/np.linalg.norm(dense_vector)
        indexid2vec[int(indexid)] = normalized_vector.squeeze()

    return indexid2vec
