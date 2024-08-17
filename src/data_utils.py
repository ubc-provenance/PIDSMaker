import os

import pickle
import torch
from torch_geometric.data import Data, TemporalData
from torch_geometric.loader import TemporalDataLoader

from encoders import TGNEncoder


def load_all_datasets(cfg):
    train_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="train")
    val_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="val")
    test_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="test")
    
    all_msg, all_t, all_edge_types = [], [], []
    for dataset in [train_data, val_data, test_data]:
        for data in dataset:
            all_msg.append(data.msg)
            all_t.append(data.t)
            all_edge_types.append(data.edge_type)

    all_msg = torch.cat(all_msg)
    all_t = torch.cat(all_t)
    all_edge_types = torch.cat(all_edge_types)
    full_data = Data(msg=all_msg, t=all_t, edge_type=all_edge_types)
    
    return train_data, val_data, test_data, full_data

def load_data_set(cfg, path: str, split: str) -> list[TemporalData]:
    """
    Returns a list of time window graphs for a given `split` (train/val/test set).
    """
    # In case we run unit tests, only some edges in the train set are present
    if cfg._test_mode:
        split = "train"

    data_list = []
    for f in sorted(os.listdir(os.path.join(path, split))):
        filepath = os.path.join(path, split, f)
        data = torch.load(filepath).to("cpu")
        data_list.append(data)

    if cfg.featurization.embed_nodes.used_method.strip() == "only_type":
        data_list = extract_msg_node_type_only(data_list, cfg)
    else:
        data_list = extract_msg_from_data(data_list, cfg)
    return data_list

def extract_msg_node_type_only(data_set: list[TemporalData], cfg) -> list[TemporalData]:
    """
    Initializes the attributes of a `Data` object based on the `msg`
    computed in previous tasks.
    """
    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types

    msg_len = data_set[0].msg.shape[1]
    expected_msg_len = (node_type_dim * 2) + edge_type_dim
    if msg_len != expected_msg_len:
        raise ValueError(f"The msg has an invalid shape, found {msg_len} instead of {expected_msg_len}")

    field_to_size = [
        ("src_type", node_type_dim),
        ("edge_type", edge_type_dim),
        ("dst_type", node_type_dim),
    ]
    for g in data_set:
        fields = {}
        idx = 0
        for field, size in field_to_size:
            fields[field] = g.msg[:, idx: idx + size]
            idx += size

        x_src = fields["src_type"]
        x_dst = fields["dst_type"]

        # If we want to predict the edge type, we remove the edge type from the message
        if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
            msg = torch.cat([x_src, x_dst], dim=-1)
        else:
            msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)
            
        edge_feats = build_edge_feats(fields, msg, cfg)

        g.x_src = x_src
        g.x_dst = x_dst
        g.msg = msg
        g.edge_type = fields["edge_type"]
        g.edge_feats = edge_feats
        g.edge_index = torch.stack([g.src, g.dst])

    return data_set

def extract_msg_from_data(data_set: list[TemporalData], cfg) -> list[TemporalData]:
    """
    Initializes the attributes of a `Data` object based on the `msg`
    computed in previous tasks.
    """
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types
    
    msg_len = data_set[0].msg.shape[1]
    expected_msg_len = (emb_dim*2) + (node_type_dim*2) + edge_type_dim
    if msg_len != expected_msg_len:
        raise ValueError(f"The msg has an invalid shape, found {msg_len} instead of {expected_msg_len}")
    
    field_to_size = [
        ("src_type", node_type_dim),
        ("src_emb", emb_dim),
        ("edge_type", edge_type_dim),
        ("dst_type", node_type_dim),
        ("dst_emb", emb_dim),
    ]
    for g in data_set:
        fields = {}
        idx = 0
        for field, size in field_to_size:
            fields[field] = g.msg[:, idx: idx + size]
            idx += size
            
        x_src = fields["src_emb"]
        x_dst = fields["dst_emb"]
        
        if cfg.detection.gnn_training.encoder.use_node_type_in_node_feats:
            x_src = torch.cat([x_src, fields["src_type"]], dim=-1)
            x_dst = torch.cat([x_dst, fields["dst_type"]], dim=-1)
        
        # If we want to predict the edge type, we remove the edge type from the message
        if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
            msg = torch.cat([x_src, x_dst], dim=-1)
        else:
            msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)
        
        edge_feats = build_edge_feats(fields, msg, cfg)
        
        g.x_src = x_src
        g.x_dst = x_dst
        g.msg = msg
        g.edge_type = fields["edge_type"]
        g.edge_feats = edge_feats
        g.edge_index = torch.stack([g.src, g.dst])
    
    return data_set

def build_edge_feats(fields, msg, cfg):
    edge_features = list(map(lambda x: x.strip(), cfg.detection.gnn_training.encoder.edge_features.split(",")))
    edge_feats = []
    if "edge_type" in edge_features:
        edge_feats.append(fields["edge_type"])
    if "msg" in edge_features:
        edge_feats.append(msg)
    edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None
    return edge_feats

def custom_temporal_data_loader(data: TemporalData, batch_size: int, *args, **kwargs):
    """
    A simple `TemporalDataLoader` which also update the edge_index with the
    sampled edges of size `batch_size`. By default, only attributes of shape (E, d)
    are updated, `edge_index` is thus not updated automatically.
    """
    loader = TemporalDataLoader(data, batch_size=batch_size, *args, **kwargs)
    for batch in loader:
        batch.edge_index = torch.stack([batch.src, batch.dst])
        yield batch

def temporal_data_to_data(data: TemporalData) -> Data:
    """
    NeighborLoader requires a `Data` object.
    We need to convert `TemporalData` to `Data` before using it.
    """
    return Data(num_nodes=data.x_src.shape[0], **{k: v for k, v in data._store.items()})

class GraphReindexer:
    """
    Simply transforms an edge_index and its src/dst node features of shape (E, d)
    to a reindexed edge_index with node IDs starting from 0 and src/dst node features of shape
    (max_num_node + 1, d).
    This reindexing is essential for the graph to be computed by a standard GNN model with PyG.
    """
    def __init__(self, num_nodes, device):
        self.num_nodes = num_nodes
        self.device = device
        
        self.assoc = None
        self.x_src_cache = None
        self.x_dst_cache = None

    def node_features_reshape(self, edge_index, x_src, x_dst, max_num_node=None):
        """
        Converts node features in shape (E, d) to a shape (N, d).
        Returns x as a tuple (x_src, x_dst).
        """
        if self.x_src_cache is None:
            self.x_src_cache = torch.zeros((self.num_nodes, x_src.shape[1]), device=self.device)
            self.x_dst_cache = torch.zeros((self.num_nodes, x_src.shape[1]), device=self.device)
            
        max_num_node = max_num_node + 1 if max_num_node else edge_index.max() + 1
        
        # To avoid storing gradients from all nodes, we detach() BEFORE caching. If we detach()
        # after storing, we loose the gradient for all operations happening before the reindexing.
        self.x_src_cache = self.x_src_cache.detach()
        self.x_dst_cache = self.x_dst_cache.detach()
        
        self.x_src_cache[edge_index[0, :]] = x_src
        self.x_dst_cache[edge_index[1, :]] = x_dst
        x = (self.x_src_cache[:max_num_node, :], self.x_dst_cache[:max_num_node, :])
        
        return x
    
    def reindex_graph(self, data):
        """
        Reindexes edge_index from 0 + reshapes node features.
        The old edge_index is stored in `data.original_edge_index`
        """
        data = data.clone()
        data.original_edge_index = data.edge_index
        (data.x_src, data.x_dst), data.edge_index = self._reindex_graph(data.edge_index, data.x_src, data.x_dst)
        return data
    
    def _reindex_graph(self, edge_index, x_src, x_dst):
        """
        Reindexes edge_index with indices starting from 0.
        Also reshapes the node features.
        """
        if self.assoc is None:
            self.assoc = torch.empty((self.num_nodes, ), dtype=torch.long, device=self.device)

        n_id = edge_index.unique()
        self.assoc[n_id] = torch.arange(n_id.size(0), device=edge_index.device)
        edge_index = self.assoc[edge_index]
        
        # Associates each feature vector to each reindexed node ID
        x = self.node_features_reshape(edge_index, x_src, x_dst)
        
        return x, edge_index

def save_model(model, path: str, cfg):
    """
    Saves only the required weights and tensors on disk.
    Using torch.save() directly on the model is very long (up to 10min),
    so we select only the tensors we want to save/load.
    """
    os.makedirs(path, exist_ok=True)
    
    # We only save specific tensors, as the other tensors are not useful to save (assoc, cache, etc)
    torch.save(model.state_dict(), os.path.join(path, "state_dict.pkl"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    
    if isinstance(model.encoder, TGNEncoder):
        torch.save(model.encoder.neighbor_loader, os.path.join(path, "neighbor_loader.pkl"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
        if cfg.detection.gnn_training.encoder.tgn.use_memory or "time_encoding" in cfg.detection.gnn_training.encoder.edge_features:
            torch.save(model.encoder.memory, os.path.join(path, "memory.pkl"), pickle_protocol=pickle.HIGHEST_PROTOCOL)

def load_model(model, path: str, cfg, map_location=None):
    """
    Loads weights and tensors from disk into a model.
    """
    model.load_state_dict(
        torch.load(os.path.join(path, "state_dict.pkl"), map_location=map_location))
    
    if isinstance(model.encoder, TGNEncoder):
        model.encoder.neighbor_loader = torch.load(os.path.join(path, "neighbor_loader.pkl"), map_location=map_location)
        if cfg.detection.gnn_training.encoder.tgn.use_memory or "time_encoding" in cfg.detection.gnn_training.encoder.edge_features:
            model.encoder.memory = torch.load(os.path.join(path, "memory.pkl"), map_location=map_location)

    return model
