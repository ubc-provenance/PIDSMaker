from sklearn.feature_extraction import FeatureHasher
import numpy as np

from config import *
from provnet_utils import *

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

def get_label2indexid(indexid2msg):
    label2indexid = {}
    for indexid, msg_list in indexid2msg.items():
        node_type = msg_list[0]
        msg = msg_list[1]
        if node_type == "file":
            spl = msg.strip().split('/')
            label ='/' + spl[1] + '/*'
            if label not in label2indexid:
                label2indexid[label] = []
            label2indexid[label].append(indexid)
    return label2indexid

def path2higlist(p):
    l=[]
    spl=p.strip().split('/')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'/'+i)
        else:
            l.append(i)
#     log(l)
    return l

def ip2higlist(p):
    l=[]
    spl=p.strip().split('.')
    for i in spl:
        if len(l)!=0:
            l.append(l[-1]+'.'+i)
        else:
            l.append(i)
#     log(l)
    return l

def list2str(l):
    s=''
    for i in l:
        s+=i
    return s

def feature_string_hashing(vec_size, indexid2msg, save_indexid2vec, save_path):
    FH_string = FeatureHasher(n_features=vec_size, input_type="string")

    indexid2vec = {}

    for indexid, msg_list in tqdm(indexid2msg.items(),desc="generating indexid2vec"):
        node_type = msg_list[0]
        msg = msg_list[1]
        if node_type == 'subject' or node_type == 'file':
            higlist = path2higlist(msg)
        else:
            higlist = ip2higlist(msg)

        higstr = list2str(higlist)
        indexid2vec[indexid] = FH_string.fit_transform([higstr]).toarray()

    if save_indexid2vec:
        torch.save(indexid2vec, save_path)
        log("indexid2vec is saved to "+save_path)

    return indexid2vec

def get_attrs_different_files(indexid2vec):
    embeddings = {}

    file_list_1 = []
    for i in range(1516491, 1516491 + 200):
        file_list_1.append(indexid2vec[i])
    file_list_1 = np.concatenate(file_list_1, axis=0)
    log(f"shape of file_list_1 is {file_list_1.shape}")
    embeddings['/data/*'] = {
        'embeddings': file_list_1,
        'color': 'r'
    }

    file_list_2 = []
    for i in range(1520698,1520897+1):
        file_list_2.append(indexid2vec[i])
    file_list_2 = np.concatenate(file_list_2, axis=0)
    log(f"shape of file_list_2 is {file_list_2.shape}")
    embeddings['/proc/*'] = {
        'embeddings': file_list_2,
        'color': 'g'
    }

    file_list_3 = []
    for i in list(range(1557820,1557855+1)) + list(range(1567860, 1567881+1)):
        file_list_3.append(indexid2vec[i])
    file_list_3 = np.concatenate(file_list_3, axis=0)
    log(f"shape of file_list_3 is {file_list_3.shape}")
    embeddings['/home/admin/*'] = {
        'embeddings': file_list_3,
        'color': 'b'
    }

    file_list_4 = []
    for i in range(1420793,1421042):
        file_list_4.append(indexid2vec[i])
    file_list_4 = np.concatenate(file_list_4, axis=0)
    log(f"shape of file_list_4 is {file_list_4.shape}")
    embeddings['/usr/share/*'] = {
        'embeddings': file_list_4,
        'color': 'yellow'
    }

    return embeddings

def build_tsne_visualization(embeddings, fig_path):
    log(f"Building TSNE visualization")
    print("Building TSNE visualization")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=100, metric='euclidean', init='pca')

    dic_2d = {}

    for key, value in embeddings.items():
        doc2vec = value['embeddings']
        embeddings_2d = tsne.fit_transform(doc2vec)

        dic_2d[key] ={
            "embeddings_2d": embeddings_2d,
            "color": value['color']
        }

    log(f"Start visualizing TSNE embeddings")
    plt.figure(figsize=(10, 8))

    for key,value in dic_2d.items():
        plt.scatter(value['embeddings_2d'][:, 0], value['embeddings_2d'][:, 1], marker='.', color=value['color'], label=key)

    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig(fig_path)
    log(f"Fig saved to {fig_path}")

def main(cfg, gen_index2vec=False):
    log_start(__file__)
    log("loading indexid2msg...")
    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur)

    fig_name = f"tsne_different_files.png"
    fig_save_dir = os.path.join(cfg.featurization.build_doc2vec._task_path, "test_HFH/")
    os.makedirs(fig_save_dir, exist_ok=True)
    fig_path = os.path.join(fig_save_dir, fig_name)

    dict_name = f"indexid2vec.pth"
    indexid2vec_path = os.path.join(fig_save_dir, dict_name)

    vec_size = cfg.featurization.build_doc2vec.vec_size

    if gen_index2vec:
        log("Start generating indexid2vec...")
        indexid2vec = feature_string_hashing(vec_size=vec_size, indexid2msg=indexid2msg, save_indexid2vec=True, save_path=indexid2vec_path)
    else:
        log("Loading indexid2vec ...")
        indexid2vec = torch.load(indexid2vec_path)

    embeddings = get_attrs_different_files(indexid2vec)
    build_tsne_visualization(embeddings, fig_path)

if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg, gen_index2vec=False)