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
from sklearn.metrics import average_precision_score, roc_auc_score
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

from config import *


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

def init_database_connection():
    if host is not None:
        connect = psycopg2.connect(database = database,
                                   host = host,
                                   user = user,
                                   password = password,
                                   port = port
                                  )
    else:
        connect = psycopg2.connect(database = database,
                                   user = user,
                                   password = password,
                                   port = port
                                  )
    cur = connect.cursor()
    return cur, connect

def gen_nodeid2msg(cur):
    # node hash id to node label and type
    sql = "select * from node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()
    nodeid2msg = {}

    # hash_id | node_type | msg | index_id
    for i in rows:
        nodeid2msg[i[0]] = i[-1]
        nodeid2msg[i[-1]] = {i[1]: i[2]}

    return nodeid2msg

def tensor_find(t,x):
    t_np=t.cpu().numpy()
    idx=np.argwhere(t_np==x)
    return idx[0][0]+1

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

def gen_darpa_rw_file(graph, walk_len, filename, adjfilename, overall_fd):
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
    # import torch
    # torch.save(adj_list,'./adj_text')
    # input()
    with open(filename,"w") as f:
        for src in tqdm(adj_list, desc="Random walking"):
            walk_num = len(adj_list[src]) * 10
            for i in range(walk_num):
                start = src
                path_sentence = ""
                for j in range(walk_len):
                    dst = random.sample(adj_list[start].keys(), 1)[0]
                    edge_type = random.sample(adj_list[start][dst], 1)[0]

                    if len(path_sentence) == 0:
                        path_sentence += f"{graph.nodes[start]['label']},{edge_type},{graph.nodes[dst]['label']}"
                    else:
                        path_sentence += f",{edge_type},{graph.nodes[dst]['label']}"

                    if dst not in adj_list:
                        break
                    else:
                        start = dst

                # We do not consider the nodes without any outgoing edges.
                # We only record the paths.
                if len(path_sentence) != 0:
                    f.write(path_sentence)
                    f.write("\n")

                    overall_fd.write(path_sentence)
                    overall_fd.write("\n")
        f.close()

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
    
    return logger
