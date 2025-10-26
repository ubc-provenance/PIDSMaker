import csv
import hashlib
import os
import random
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from time import mktime

import networkx as nx
import nltk
import numpy as np
import psycopg2
import pytz
import torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nltk.download("punkt", quiet=True)

from pidsmaker.config import update_cfg_for_multi_dataset


def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()


def ns_time_to_datetime(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    dt = datetime.fromtimestamp(int(ns) // 1000000000)
    s = dt.strftime("%Y-%m-%d %H:%M:%S")
    s += "." + str(int(int(ns) % 1000000000)).zfill(9)
    return s


def ns_time_to_datetime_US(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    tz = pytz.timezone("US/Eastern")
    dt = pytz.datetime.datetime.fromtimestamp(int(ns) // 1000000000, tz)
    s = dt.strftime("%Y-%m-%d %H:%M:%S")
    s += "." + str(int(int(ns) % 1000000000)).zfill(9)
    return s


def time_to_datetime_US(s):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00
    """
    tz = pytz.timezone("US/Eastern")
    dt = pytz.datetime.datetime.fromtimestamp(int(s), tz)
    s = dt.strftime("%Y-%m-%d %H:%M:%S")

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
    tz = pytz.timezone("US/Eastern")
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
    tz = pytz.timezone("US/Eastern")
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp
    return int(timeStamp)

def OPTC_datetime_to_timestamp_US(date):
    '''convert OPTC datetime string to timestamp in nanoseconds'''
    date=date.replace('-04:00','')
    if '.' in date:
        date,ms=date.split('.')
    else:
        ms=0
    tz = pytz.timezone('Etc/GMT+4')
    timeArray = time.strptime(date, "%Y-%m-%dT%H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp=timestamp.timestamp()
    timeStamp = timestamp*1000+int(ms)
    return int(timeStamp) * 1000000

def init_database_connection(cfg):
    if cfg.preprocessing.build_graphs.use_all_files:
        database_name = cfg.dataset.database_all_file
    else:
        database_name = cfg.dataset.database

    if cfg.database.host is not None:
        connect = psycopg2.connect(
            database=database_name,
            host=cfg.database.host,
            user=cfg.database.user,
            password=cfg.database.password,
            port=cfg.database.port,
        )
    else:
        connect = psycopg2.connect(
            database=database_name,
            user=cfg.database.user,
            password=cfg.database.password,
            port=cfg.database.port,
        )
    cur = connect.cursor()
    return cur, connect


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


def gen_darpa_rw_file(walk_len, corpus_fd, adjfilename, overall_fd, num_walks=10):
    adj_list = {}
    back_adj_list = {}
    with open(adjfilename, "r") as adj_file:
        for line in log_tqdm(adj_file, desc="Creating adj list"):
            line = line.strip().split(",")
            srcID = line[0]
            dstID = line[1]
            edgeLabel = line[4]

            if srcID not in adj_list:
                adj_list[srcID] = {}

            if dstID not in adj_list[srcID]:
                adj_list[srcID][dstID] = set()

            if dstID not in back_adj_list:
                back_adj_list[dstID] = {}

            if srcID not in back_adj_list[dstID]:
                back_adj_list[dstID][srcID] = set()

            adj_list[srcID][dstID].add(edgeLabel)
            back_adj_list[dstID][srcID].add(edgeLabel)

    if True:
        lines_buffer = []

        # Computing random neighbors for each iteration is extremely slow.
        # We thus pre-compute a list of random indices for all unique numbers of neighbors.
        # These indices can then be accessed given the length of the neighbors.
        unique_neighbors_count = set(len(v) for v in adj_list.values()) | set(
            len(v) for v in back_adj_list.values()
        )
        cache_size = 10 * len(adj_list) * num_walks * walk_len
        random_cache = {
            count: np.random.randint(0, count, size=cache_size) for count in unique_neighbors_count
        }
        random_idx = {count: 0 for count in unique_neighbors_count}

        def get_rand(count):
            if count not in random_cache:
                return np.random.randint(0, count)
            idx = random_idx[count]
            val = random_cache[count][idx]
            random_idx[count] = (idx + 1) % cache_size  # Wrap-around cache
            return val

        for src in log_tqdm(adj_list, desc="Forward random walking"):
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
                        # path_sentence.append(f"{graph.nodes[start]['label']},{edge_type},{graph.nodes[dst]['label']}")
                        path_sentence.append(f"{start},{edge_type},{dst}")
                    else:
                        # path_sentence.append(f"{edge_type},{graph.nodes[dst]['label']}")
                        path_sentence.append(f"{edge_type},{dst}")

                    if dst not in adj_list:
                        break
                    else:
                        start = dst

                # We do not consider the nodes without any outgoing edges.
                # We only record the paths.
                if len(path_sentence) != 0:
                    path_str = ",".join(path_sentence)
                    lines_buffer.append(path_str)

        # Run bidirectional random walking to ensure that every node appears in the corpus
        # Missing sink nodes leads to issues when training A La Carte Matrix
        for dst in log_tqdm(back_adj_list, desc="Backward random walking"):
            walk_num = len(back_adj_list[dst]) * num_walks  # NOTE: it seems a lot
            for i in range(walk_num):
                start = dst
                path_sentence = []
                for j in range(walk_len):
                    start_keys = list(back_adj_list[start].keys())
                    src = start_keys[get_rand(len(start_keys))]

                    start_src = list(back_adj_list[start][src])
                    edge_type = start_src[get_rand(len(start_src))]

                    if len(path_sentence) == 0:
                        # path_sentence.append(f"{graph.nodes[start]['label']},{edge_type},{graph.nodes[dst]['label']}")
                        path_sentence.append(f"{start},{edge_type},{src}")
                    else:
                        # path_sentence.append(f"{edge_type},{graph.nodes[dst]['label']}")
                        path_sentence.append(f"{edge_type},{src}")

                    if src not in back_adj_list:
                        break
                    else:
                        start = src

                # We do not consider the nodes without any outgoing edges.
                # We only record the paths.
                if len(path_sentence) != 0:
                    path_str = ",".join(path_sentence)
                    lines_buffer.append(path_str)

        lines = "\n".join(lines_buffer)
        corpus_fd.write(lines)
        overall_fd.write(lines)


def gen_darpa_adj_files(graph, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        for u, v, k in graph.edges:
            # srcID, dstID, srcLabel, dstLabel, edgeLabel
            data = [
                u,
                v,
                graph.nodes[u]["label"],
                graph.nodes[v]["label"],
                graph.edges[u, v, k]["label"] if "label" in graph.edges[u, v, k] else "",
                graph.nodes[u]["node_type"],
                graph.nodes[v]["node_type"],
            ]
            writer.writerow(data)
        f.close()


def log_start(_file_: str):
    log(f"======= START SUBTASK {os.path.basename(_file_)} =======")


def get_all_files_from_folders(base_dir: str, folders: list[str]):
    paths = [
        os.path.abspath(os.path.join(base_dir, sub, f))
        for sub in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, sub)) and sub in folders
        for f in os.listdir(os.path.join(base_dir, sub))
    ]
    paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return paths

def get_all_filelist(filepath):
    '''get all file paths under the given filepath recursively'''
    file_paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            full_path = os.path.join(root, file)
            abs_path = os.path.abspath(full_path)
            file_paths.append(abs_path)
    return file_paths

def load_graphs_for_days(base_dir, days):
    """Loads all graph snapshots for a given list of days."""
    return [
        torch.load(path) for day in days for path in get_all_files_from_folders(base_dir, [day])
    ]


def listdir_sorted(path: str):
    files = os.listdir(path)
    files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))  # sorted by ascending number
    return files


def remove_underscore_keys(data, keys_to_keep=[], keys_to_rm=[]):
    for key in list(data.keys()):
        if (key in keys_to_rm) or (key.startswith("_") and key not in keys_to_keep):
            del data[key]
        elif isinstance(data[key], dict):
            data[key] = dict(data[key])
            remove_underscore_keys(data[key], keys_to_keep, keys_to_rm)
    return data


def tokenize_subject(sentence: str):
    new_sentence = re.sub(r"\\+", "/", sentence)
    return word_tokenize(new_sentence.replace("/", " / "))
    # return word_tokenize(sentence.replace('/',' ').replace('=',' = ').replace(':',' : '))


def tokenize_file(sentence: str):
    new_sentence = re.sub(r"\\+", "/", sentence)
    return word_tokenize(new_sentence.replace("/", " / "))


def tokenize_netflow(sentence: str):
    return word_tokenize(sentence.replace(":", " : ").replace(".", " . "))


def tokenize_label(node_label, node_type):
    if node_label == "":
        return [""]
    elif node_type == "subject":
        return tokenize_subject(node_label)
    elif node_type == "file":
        return tokenize_file(node_label)
    elif node_type == "netflow":
        return tokenize_netflow(node_label)
    raise ValueError("Invalid node type")


def tokenize_arbitrary_label(sentence):
    new_sentence = re.sub(r"\\+", "/", sentence)
    return word_tokenize(new_sentence.replace("/", " / ").replace(":", " : ").replace(".", " . "))


def log(msg: str, return_line=False, pre_return_line=False, *args, **kwargs):
    if pre_return_line:
        print("")

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {msg}", *args, **kwargs)

    if return_line:
        print("")


DISABLE_TQDM = True


def log_tqdm(iterator, desc="", logging=True, **kwargs):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    if DISABLE_TQDM and logging:
        log(f"{desc}...")
    return tqdm(iterator, desc=f"{timestamp} - {desc}", disable=DISABLE_TQDM, **kwargs)


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
            "subject": "SELECT index_id, path, cmd FROM subject_node_table;",
        }
        node_to_path_type = {}
        for node_type, query in queries.items():
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                if node_type == "netflow":
                    index_id, src_addr, dst_addr, src_port, dst_port = row
                    node_to_path_type[index_id] = {
                        "path": f"{str(src_addr)}:{str(src_port)}->{str(dst_addr)}:{str(dst_port)}",
                        "type": node_type,
                    }
                elif node_type == "file":
                    index_id, path = row
                    node_to_path_type[index_id] = {"path": str(path), "type": node_type}
                elif node_type == "subject":
                    index_id, path, cmd = row
                    node_to_path_type[index_id] = {"path": str(path), "type": node_type, "cmd": cmd}

        torch.save(node_to_path_type, out_file)
        connect.close()

    else:
        node_to_path_type = torch.load(out_file)

    return node_to_path_type


def copy_directory(src_path, dest_path):
    if not os.path.isdir(src_path):
        log(f"The source path '{src_path}' does not exist or is not a directory.")
        return

    if os.path.exists(dest_path):
        log(f"The destination path '{dest_path}' already exists. Removing it for a fresh copy.")
        shutil.rmtree(dest_path)

    try:
        shutil.copytree(src_path, dest_path)
        log(f"Directory copied successfully from '{src_path}' to '{dest_path}'.")
    except Exception as e:
        log(f"An error occurred while copying the directory: {e}")


def get_split_to_files(cfg, base_dir):
    return {
        "train": get_all_files_from_folders(base_dir, cfg.dataset.train_files),
        "val": get_all_files_from_folders(base_dir, cfg.dataset.val_files),
        "test": get_all_files_from_folders(base_dir, cfg.dataset.test_files),
    }


def gen_relation_onehot(rel2id):
    relvec = torch.nn.functional.one_hot(
        torch.arange(0, len(rel2id.keys()) // 2), num_classes=len(rel2id.keys()) // 2
    )
    rel2vec = {}
    for i in rel2id.keys():
        if type(i) is not int:
            rel2vec[i] = relvec[rel2id[i] - 1]
            rel2vec[relvec[rel2id[i] - 1]] = i
    return rel2vec


def get_indexid2msg(cfg, gather_multi_dataset=False):
    def load_file(cfg):
        indexid2msg_file = os.path.join(
            cfg.preprocessing.build_graphs._dicts_dir, "indexid2msg.pkl"
        )
        indexid2msg = torch.load(indexid2msg_file)
        return indexid2msg

    if gather_multi_dataset:
        multi_datasets = get_multi_datasets(cfg)
        log(f"Multi-dataset order: {multi_datasets}")

        all_indexid2msg = {}
        cumsum = 0
        for dataset in multi_datasets:
            cfg, _ = update_cfg_for_multi_dataset(cfg, dataset)

            indexid2msg = load_file(cfg)
            max_node = max(map(int, list(indexid2msg.keys())))
            indexid2msg = {str(int(k) + cumsum): v for k, v in indexid2msg.items()}
            cumsum += max_node

            # Merge the cumsumed indices and original values from all datasets
            all_indexid2msg = {**all_indexid2msg, **indexid2msg}
        return all_indexid2msg

    indexid2msg = load_file(cfg)
    indexid2msg = dict(sorted(indexid2msg.items(), key=lambda item: int(item[0])))
    return indexid2msg


def get_split2nodes(cfg, gather_multi_dataset=False):
    def load_file(cfg):
        path = os.path.join(cfg.preprocessing.build_graphs._dicts_dir, "split2nodes.pkl")
        split2nodes = torch.load(path)
        return split2nodes

    multi_datasets = get_multi_datasets(cfg)
    use_multi_dataset = "none" not in cfg.preprocessing.build_graphs.multi_dataset

    if gather_multi_dataset and use_multi_dataset:
        all_split2nodes = defaultdict(set)
        cumsum = 0
        for dataset in multi_datasets:
            cfg, _ = update_cfg_for_multi_dataset(cfg, dataset)

            split2nodes = load_file(cfg)
            indexid2msg = get_indexid2msg(cfg)
            max_node = max(map(int, list(indexid2msg.keys())))
            split2nodes = {k: {str(int(e) + cumsum) for e in v} for k, v in split2nodes.items()}
            cumsum += max_node

            for k, v in split2nodes.items():
                all_split2nodes[k] |= v
        return all_split2nodes

    return load_file(cfg)


def compute_class_weights(labels, num_classes):
    """
    Compute balanced class weights for a given set of labels using PyTorch,
    and pad the weights with 0s if some classes are missing in the batch.

    Parameters:
        labels (Tensor): A 1D tensor containing class indices for each sample.
        num_classes (int): The number of unique classes.

    Returns:
        Tensor: A tensor containing the weight for each class, padded to num_classes.
    """
    # From 1-hot to label index
    labels = labels.argmax(dim=-1)

    # Count the number of instances for each class
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    total_count = len(labels)

    # Initialize weights with zeros for all classes
    class_weights = torch.zeros(num_classes, device=labels.device)

    # Avoid division by zero by checking for non-zero class counts
    non_zero_classes = class_counts > 0
    class_weights[non_zero_classes] = total_count / (num_classes * class_counts[non_zero_classes])

    return class_weights


def calculate_average_from_file(filename):
    numbers = []
    try:
        with open(filename, "r") as f:
            for line in f:
                numbers.append(float(line.strip()))
        if numbers:
            return sum(numbers) / len(numbers)
        else:
            return 1e-9
    except FileNotFoundError:
        log(f"{filename} does not exist")
        return None


def get_events_between_GPs(cur, start_time, end_time, malicious_nodes: list):
    malicious_nodes_str = ", ".join(f"'{node}'" for node in malicious_nodes)
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}' AND src_index_id IN ({malicious_nodes_str}) AND dst_index_id IN ({malicious_nodes_str});"
    cur.execute(sql)
    rows = cur.fetchall()
    return rows


def get_events_between_time_range(
    cur,
    start_time,
    end_time,
):
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}';"
    cur.execute(sql)
    rows = cur.fetchall()
    return rows


def generate_DAG(edges):
    node_version = {}
    for u, v, t in edges:
        if u not in node_version:
            node_version[u] = 0
        if v not in node_version:
            node_version[v] = 0

    sorted_edges = sorted(edges, key=lambda x: x[2])

    new_nodes = set()
    new_edges = []
    visited = set()
    for u, v, t in sorted_edges:
        if u == v:
            continue

        src = str(u) + "-" + str(node_version[u])
        visited.add(u)
        new_nodes.add(src)

        if v not in visited:
            dst = str(v) + "-" + str(node_version[v])
            visited.add(v)
            new_nodes.add(dst)
            new_edges.append((src, dst, {"time": int(t)}))
        else:
            dst_current = str(v) + "-" + str(node_version[v])
            dst_new = str(v) + "-" + str(node_version[v] + 1)
            node_version[v] += 1
            new_nodes.add(dst_new)
            new_edges.append((src, dst_new, {"time": int(t)}))
            new_edges.append((dst_current, dst_new, {"time": int(t)}))

    DAG = nx.DiGraph()
    DAG.add_nodes_from(list(new_nodes))
    DAG.add_edges_from(new_edges)

    return DAG, node_version


def log_dataset_stats(datasets):
    def log_helper(label, dataset):
        edges = torch.tensor([d.src.shape[0] for d in dataset])
        nodes = torch.tensor([torch.unique(d.edge_index).shape[0] for d in dataset])

        log(f"{label} num graphs: {len(dataset)}")
        log(
            f"{label} edges | mean: {int(torch.mean(edges, dtype=torch.float))} | min: {int(torch.min(edges))} | max: {int(torch.max(edges))}"
        )
        log(
            f"{label} nodes | mean: {int(torch.mean(nodes, dtype=torch.float))} | min: {int(torch.min(nodes))} | max: {int(torch.max(nodes))}"
        )
        log("")

    train_data, val_data, test_data = datasets
    log("")
    log("Dataset statistics")
    for train_graphs, val_graphs in zip(train_data, val_data):
        for label, dataset in [("Train", train_graphs), ("Val", val_graphs)]:
            log_helper(label, dataset)

    for test_graphs in test_data:
        for label, dataset in [("Test", test_graphs)]:
            log_helper(label, dataset)


def set_seed(cfg):
    if cfg.detection.gnn_training.use_seed:
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    if cfg.detection.gnn_training.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_multi_datasets(cfg):
    # The main dataset should be always the first one
    multi_dataset = cfg.preprocessing.build_graphs.multi_dataset
    multi_datasets = list(
        map(lambda x: x.strip(), multi_dataset.split("," if "," in multi_dataset else "-"))
    )
    return [cfg.dataset.name] + [d for d in multi_datasets if d != cfg.dataset.name]
