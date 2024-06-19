import pytz
from time import mktime
from datetime import datetime
import time
import psycopg2
from psycopg2 import extras as ex
import os.path as osp
import os
import copy
import logging
import torch
from torch.nn import Linear
from torch_geometric.data import TemporalData
from torch_geometric.nn import TGNMemory, TransformerConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage, MeanAggregator,
                                           LastAggregator)
from torch_geometric import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import math
import copy
import time
import xxhash
import gc
import random
import csv
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score as ap_score,
)

from config import *
import hashlib
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256() # TODO: check if we might remove it in the future
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def ns_time_to_datetime(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    dt = datetime.fromtimestamp(int(ns) // 1000000000)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def ns_time_to_datetime_US(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(int(int(ns) % 1000000000)).zfill(9)
    return s

def time_to_datetime_US(s):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00
    """
    tz = pytz.timezone('US/Eastern')
    dt = pytz.datetime.datetime.fromtimestamp(int(s), tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')

    return s

def datetime_to_ns_time(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    timeStamp = timeStamp * 1000000000
    return timeStamp

def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)

def datetime_to_timestamp_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp
    return int(timeStamp)

def init_database_connection(cfg):
    if cfg.preprocessing.build_graphs.use_all_files:
        database_name = cfg.dataset.database_all_file
    else:
        database_name = cfg.dataset.database

    if cfg.database.host is not None:
        connect = psycopg2.connect(database = database_name,
                                   host = cfg.database.host,
                                   user = cfg.database.user,
                                   password = cfg.database.password,
                                   port = cfg.database.port
                                  )
    else:
        connect = psycopg2.connect(database = database_name,
                                   user = cfg.database.user,
                                   password = cfg.database.password,
                                   port = cfg.database.port
                                  )
    cur = connect.cursor()
    return cur, connect

def gen_nodeid2msg(cur, use_cmd=True, use_port=False):
    # node hash id to node label and type
    # {hash_id: index_id} and {index_id: {node_type:msg}}
    indexid2msg = {}

    # netflow
    sql = """
        select * from netflow_node_table;
        """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        hash_id = i[1]
        remote_ip = i[4]
        remote_port = i[5]
        index_id = i[-1] # int
        indexid2msg[hash_id] = index_id
        if use_port:
            indexid2msg[index_id] = {'netflow': remote_ip + ':' +remote_port}
        else:
            indexid2msg[index_id] = {'netflow': remote_ip}

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        hash_id = i[1]
        path = i[2]
        cmd = i[3]
        index_id = i[-1]
        indexid2msg[hash_id] = index_id
        if use_cmd:
            indexid2msg[index_id] = {'subject': path + ' ' +cmd}
        else:
            indexid2msg[index_id] = {'subject': path}

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    for i in records:
        hash_id = i[1]
        path = i[2]
        index_id = i[-1]
        indexid2msg[hash_id] = index_id
        indexid2msg[index_id] = {'file': path}

    return indexid2msg #{hash_id: index_id} and {index_id: {node_type:msg}}

def std(t):
    t = np.array(t)
    return np.std(t)

def var(t):
    t = np.array(t)
    return np.var(t)

def mean(t):
    t = np.array(t)
    return np.mean(t)

def percentile_90(t):
    sorted_data = np.sort(t)
    Q = np.percentile(sorted_data, 90)
    return Q

def percentile_75(t):
    sorted_data = np.sort(t)
    Q = np.percentile(sorted_data, 75)
    return Q

def percentile_50(t):
    sorted_data = np.sort(t)
    Q = np.percentile(sorted_data, 50)
    return Q

def hashgen(l):
    """Generate a single hash value from a list. @l is a list of
    string values, which can be properties of a node/edge. This
    function returns a single hashed integer value."""
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()


def split_filename(path):
    '''
    Given a path, split it based on '/' and file extension.
    e.g.
        "/home/test/Desktop/123.txt" => "home test Desktop 123 txt"
    :param path: the name of the path
    :return: the split path name
    '''
    file_name, file_extension = os.path.splitext(os.path.basename(path))
    file_extension = file_extension.replace(".","")
    result = ' '.join(path.split('/')[1:-1]) + ' ' + file_name + ' ' + file_extension
    return result

def gen_darpa_rw_file(graph, walk_len, filename, adjfilename, overall_fd, num_walks=10):
    adj_list = {}
    with open(adjfilename, 'r') as adj_file:
        for line in tqdm(adj_file, desc="creating adj list"):
            line = line.strip().split(",")
            srcID = line[0]
            dstID = line[1]
            edgeLabel = line[4]

            if srcID not in adj_list:
                adj_list[srcID] = {}

            if dstID not in adj_list[srcID]:
                adj_list[srcID][dstID] = set()

            adj_list[srcID][dstID].add(edgeLabel)

    with open(filename, "w") as f:
        lines_buffer = []

        # Computing random neighbors for each iteration is extremely slow.
        # We thus pre-compute a list of random indices for all unique numbers of neighbors.
        # These indices can then be accessed given the length of the neighbors.
        unique_neighbors_count = list(set([len(v) for k, v in adj_list.items()]))
        cache_size = 5 * len(adj_list) * num_walks * walk_len
        random_cache = {count: np.random.randint(0, count, size=cache_size) for count in unique_neighbors_count}
        random_idx = {count: 0 for count in unique_neighbors_count}

        def get_rand(idx: int):
            try:
                val = random_cache[idx][random_idx[idx]]
                random_idx[idx] += 1
            except KeyError:
                return np.random.randint(0, idx)
            return val

        for src in tqdm(adj_list, desc="Random walking"):
            walk_num = len(adj_list[src]) * num_walks
            for i in range(walk_num):
                start = src
                path_sentence = []
                for j in range(walk_len):
                    start_keys = list(adj_list[start].keys())
                    dst = start_keys[get_rand(len(start_keys))]

                    start_dst = list(adj_list[start][dst])
                    edge_type = start_dst[get_rand(len(start_dst))]

                    if len(path_sentence) == 0:
                        path_sentence.append(f"{graph.nodes[start]['label']},{edge_type},{graph.nodes[dst]['label']}")
                    else:
                        path_sentence.append(f"{edge_type},{graph.nodes[dst]['label']}")

                    if dst not in adj_list:
                        break
                    else:
                        start = dst

                # We do not consider the nodes without any outgoing edges.
                # We only record the paths.
                if len(path_sentence) != 0:
                    path_str = ",".join(path_sentence)
                    lines_buffer.append(path_str)
        
        lines = "\n".join(lines_buffer)
        f.write(lines)
        overall_fd.write(lines)

def gen_darpa_adj_files(graph, filename):
    with open(filename,"w") as f:
        writer = csv.writer(f)
        for (u,v,k) in graph.edges:
            # srcID, dstID, srcLabel, dstLabel, edgeLabel
            data = [
                u,
                v,
                graph.nodes[u]["label"],
                graph.nodes[v]["label"],
                graph.edges[u,v,k]["label"],
                graph.nodes[u]["node_type"],
                graph.nodes[v]["node_type"]
            ]
            writer.writerow(data)
        f.close()

def get_logger(name: str, filename: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("")
    logger.info(f"START LOGGING FOR SUBTASK: {name}")
    logger.info("")
    
    log("")
    log(f"START LOGGING FOR SUBTASK: {name}")
    log("")
    
    return logger

def get_all_files_from_folders(base_dir: str, folders: list[str]):
    return sorted([os.path.abspath(os.path.join(base_dir, sub, f))
        for sub in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, sub)) and sub in folders
        for f in os.listdir(os.path.join(base_dir, sub))])

def listdir_sorted(path: str):
    files = os.listdir(path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) # sorted by ascending number
    return files

def remove_underscore_keys(data, keys_to_keep=[], keys_to_rm=[]):
    for key in list(data.keys()):
        if (key in keys_to_rm) or (key.startswith('_') and key not in keys_to_keep):
            del data[key]
        elif isinstance(data[key], dict):
            data[key] = dict(data[key])
            remove_underscore_keys(data[key], keys_to_keep, keys_to_rm)
    return data

def classifier_evaluation(y_test, y_test_pred, scores):
    labels_exist = sum(y_test) > 0
    if labels_exist:
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    else:
        tn, fp, fn, tp = 1, 1, 1, 1  # only to not break tests

    fpr = fp/(fp+tn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    fscore=2*(precision*recall)/(precision+recall)

    try:
        auc_val=roc_auc_score(y_test, scores)
    except: auc_val=float("nan")
    try:
        ap=ap_score(y_test, scores)
    except: ap=float("nan")
    
    log(f'total num: {len(y_test)}')
    log(f'tn: {tn}')
    log(f'fp: {fp}')
    log(f'fn: {fn}')
    log(f'tp: {tp}')
    log('')

    log(f"ap: {ap}")
    log(f"precision: {precision}")
    log(f"recall: {recall}")
    log(f"fpr: {fpr}")
    log(f"fscore: {fscore}")
    log(f"accuracy: {accuracy}")
    log(f"auc_val: {auc_val}")

    stats = {
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "fpr": round(fpr, 7),
        "fscore": round(fscore, 5),
        "ap": round(ap, 5),
        "accuracy": round(accuracy, 5),
        "auc_val": round(auc_val, 5),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    return stats

def get_indexid2msg(cur, use_cmd=True, use_port=False):
    indexid2msg = {}

    # netflow
    sql = """
        select * from netflow_node_table;
        """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of netflow nodes: {len(records)}")

    for i in records:
        remote_ip = i[4]
        remote_port = i[5]
        index_id = i[-1] # int
        if use_port:
            indexid2msg[index_id] = ['netflow', remote_ip + ':' +remote_port]
        else:
            indexid2msg[index_id] = ['netflow', remote_ip]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of process nodes: {len(records)}")

    for i in records:
        path = i[2]
        cmd = i[3]
        index_id = i[-1]
        if use_cmd:
            indexid2msg[index_id] = ['subject', path + ' ' +cmd]
        else:
            indexid2msg[index_id] = ['subject', path]

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()

    log(f"Number of file nodes: {len(records)}")

    for i in records:
        path = i[2]
        index_id = i[-1]
        indexid2msg[index_id] = ['file', path]

    return indexid2msg #{index_id: [node_type, msg]}

def tokenize_subject(sentence: str):
    return word_tokenize(sentence.replace('/', ' / '))
    # return word_tokenize(sentence.replace('/',' ').replace('=',' = ').replace(':',' : '))
def tokenize_file(sentence: str):
    return word_tokenize(sentence.replace('/',' / '))
def tokenize_netflow(sentence: str):
    return word_tokenize(sentence.replace(':',' : ').replace('.',' . '))

def log(msg: str, *args):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {msg}", *args)

def get_device(cfg):
    if cfg._use_cpu:
        return torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        log("Warning: the device is CPU instead of CUDA")
    return device

def get_node_to_path_and_type(cfg):
    out_path = cfg.preprocessing.build_graphs._node_id_to_path
    out_file = os.path.join(out_path, "node_to_paths.pkl")
    
    if not os.path.exists(out_file):
        os.makedirs(out_path, exist_ok=True)
        cur, connect = init_database_connection(cfg)
        
        queries = {
            "file": "SELECT index_id, path FROM file_node_table;",
            "netflow": "SELECT index_id, src_addr, dst_addr, src_port, dst_port FROM netflow_node_table;",
            "subject": "SELECT index_id, path FROM subject_node_table;"
        }
        node_to_path_type = {}
        for node_type, query in queries.items():
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                if node_type == "netflow":
                    index_id, src_addr, dst_addr, src_port, dst_port = row
                    node_to_path_type[index_id] = {"path": f"{src_addr}:{src_port}->{dst_addr}:{dst_port}", "type": node_type}
                else:
                    index_id, path = row
                    node_to_path_type[index_id] = {"path": path, "type": node_type}

        torch.save(node_to_path_type, out_file)
        connect.close()
        
    else:
        node_to_path_type = torch.load(out_file)
        
    return node_to_path_type
