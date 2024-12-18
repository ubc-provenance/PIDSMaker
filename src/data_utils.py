import math
import os

import pickle
import torch
from torch_geometric.data import Data, TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data.collate import collate
from torch_geometric.data.data import size_repr
from torch_scatter import scatter

from encoders import TGNEncoder

class CollatableTemporalData(TemporalData):
    """
    We use this class instead of TemporalData in order to easily concatenate data
    objects together without any batching behavior.
    Normal TemporalData doesn't support edge_index so we define it here.
    """
    def __init__(
        self,
        src=None,
        dst=None,
        t=None,
        msg=None,
        **kwargs,
    ):
        super().__init__(src=src, dst=dst, t=t, msg=msg, **kwargs)
        
    def __inc__(self, key: str, value, *args, **kwargs):
        return 0

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        return 1 if "index" in key else 0
    
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = ', '.join([size_repr(k, v) for k, v in self._store.items()])
        info += ", " + size_repr("edge_index", self.edge_index)
        return f'{cls}({info})'

def load_all_datasets(cfg):
    train_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="train")
    val_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="val")
    test_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="test")
    
    all_msg, all_t, all_edge_types = [], [], []
    max_node = 0
    for dataset in [train_data, val_data, test_data]:
        for data in dataset:
            all_msg.append(data.msg)
            all_t.append(data.t)
            all_edge_types.append(data.edge_type)
            max_node = max(max_node, torch.cat([data.src, data.dst]).max().item())

    if "tgn" in cfg.detection.gnn_training.encoder.used_methods:
        all_msg = torch.cat(all_msg)
        all_t = torch.cat(all_t)
        all_edge_types = torch.cat(all_edge_types)
        full_data = Data(msg=all_msg, t=all_t, edge_type=all_edge_types)
    else:
        full_data = None
    max_node = max_node + 1
    print(f"Max node in {cfg.dataset.name}: {max_node}")
    
    # Concatenates all data into a single data so that iterating over batches
    # of edges is more consistent with TGN
    batch_mode = cfg.detection.gnn_training.batch_mode
    batch_size = cfg.detection.gnn_training.edge_batch_size
    if batch_size not in [None, 0]:
        train_data = batch_temporal_data(collate_temporal_data(train_data), batch_size, batch_mode, cfg)
        val_data = batch_temporal_data(collate_temporal_data(val_data), batch_size, batch_mode, cfg)
        test_data = batch_temporal_data(collate_temporal_data(test_data), batch_size, batch_mode, cfg)
    
    return train_data, val_data, test_data, full_data, max_node

def load_data_set(cfg, path: str, split: str) -> list[CollatableTemporalData]:
    """
    Returns a list of time window graphs for a given `split` (train/val/test set).
    """

    data_list = []
    for f in sorted(os.listdir(os.path.join(path, split))):
        filepath = os.path.join(path, split, f)
        data = torch.load(filepath).to("cpu")
        data_list.append(data)

    data_list = extract_msg_from_data(data_list, cfg)
    return data_list

def extract_msg_from_data(data_set: list[CollatableTemporalData], cfg) -> list[CollatableTemporalData]:
    """
    Initializes the attributes of a `Data` object based on the `msg`
    computed in previous tasks.
    """
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    only_type = cfg.featurization.embed_nodes.used_method.strip() == "only_type"
    if only_type or emb_dim is None:
        emb_dim = 0
    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types
    selected_node_feats = cfg.detection.gnn_training.encoder.node_features
    
    msg_len = data_set[0].msg.shape[1]
    expected_msg_len = (emb_dim*2) + (node_type_dim*2) + edge_type_dim
    if msg_len != expected_msg_len:
        raise ValueError(f"The msg has an invalid shape, found {msg_len} instead of {expected_msg_len}")
    
    field_to_size = {
        "src_type": node_type_dim,
        "src_emb": emb_dim,
        "edge_type": edge_type_dim,
        "dst_type": node_type_dim,
        "dst_emb": emb_dim,
    }
    
    if "edges_distribution" in selected_node_feats:
        max_num_nodes = max([torch.cat([g.src, g.dst]).max().item() for g in data_set]) + 1
        x_distrib = torch.zeros(max_num_nodes, edge_type_dim * 2, dtype=torch.float)
        
    if only_type:
        selected_node_feats = ["node_type"]
    else:
        selected_node_feats = list(map(lambda x: x.strip(), selected_node_feats.replace("-", ",").split(",")))
    
    for g in data_set:
        fields = {}
        idx = 0
        for field, size in field_to_size.items():
            fields[field] = g.msg[:, idx: idx + size]
            idx += size
            
        # Selects only the node features we want
        x_src, x_dst = [], []
        for feat in selected_node_feats:
        
            if feat == "node_emb":
                x_src.append(fields["src_emb"])
                x_dst.append(fields["dst_emb"])
        
            elif feat == "node_type":
                x_src.append(fields["src_type"])
                x_dst.append(fields["dst_type"])
                
            elif feat == "edges_distribution": # as in ThreaTrace
                x_distrib.scatter_add_(0, g.src.unsqueeze(1).expand(-1, edge_type_dim), fields["edge_type"])
                x_distrib[:, edge_type_dim:].scatter_add_(0, g.dst.unsqueeze(1).expand(-1, edge_type_dim), fields["edge_type"])
                
                # In ThreaTrace they don't standardize, here we do standardize by max value in TW
                x_distrib = x_distrib / (x_distrib.max() + 1e-12)
                
                x_src.append(x_distrib[g.src])
                x_dst.append(x_distrib[g.dst])
                
                x_distrib.fill_(0)
                
            else:
                raise ValueError(f"Node feature {feat} is invalid.")
            
        x_src = torch.cat(x_src, dim=-1)
        x_dst = torch.cat(x_dst, dim=-1)
        
        # If we want to predict the edge type, we remove the edge type from the message
        if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
            msg = torch.cat([x_src, x_dst], dim=-1)
        else:
            msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)
        
        edge_feats = build_edge_feats(fields, msg, cfg)
        
        g.x_src = x_src
        g.x_dst = x_dst
        g.edge_type = fields["edge_type"]
        g.edge_feats = edge_feats
        
        if "tgn" in cfg.detection.gnn_training.encoder.used_methods and cfg.detection.gnn_training.encoder.tgn.use_memory:
            g.msg = msg
        
        # NOTE: do not add edge_index as it is already within `CollatableTemporalData`
        # g.edge_index = ...
        
        if cfg._is_node_level:
            g.node_type_src = fields["src_type"]
            g.node_type_dst = fields["dst_type"]
    
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

def custom_temporal_data_loader(data: CollatableTemporalData, batch_size: int, *args, **kwargs):
    """
    A simple `TemporalDataLoader` which also update the edge_index with the
    sampled edges of size `batch_size`. By default, only attributes of shape (E, d)
    are updated, `edge_index` is thus not updated automatically.
    """
    loader = TemporalDataLoader(data, batch_size=batch_size, *args, **kwargs)
    for batch in loader:
        yield batch

def temporal_data_to_data(data: CollatableTemporalData) -> Data:
    """
    NeighborLoader requires a `Data` object.
    We need to convert `CollatableTemporalData` to `Data` before using it.
    """
    data = Data(num_nodes=data.x_src.shape[0], **{k: v for k, v in data._store.items()})
    del data.num_nodes
    return data

def collate_temporal_data(data_list: list[CollatableTemporalData]) -> CollatableTemporalData:
    """
    Concatenates attributes from data ojects into a single data object.
    Do not use with `Data` directly because it will use batching when collating.
    """
    assert all([not isinstance(data, Data) for data in data_list]), "Concatenating Data objects result in batching."
    
    data = collate(CollatableTemporalData, data_list)[0]
    del data.ptr
    del data.batch

    return data

def batch_temporal_data(data: CollatableTemporalData, batch_size: int, batch_mode: str, cfg) -> list[CollatableTemporalData]:
    if batch_mode == "edges":
        num_batches = math.ceil(len(data.src) / batch_size)  # NOTE: the last batch won't have the same number of edges as the batch
        
        data_list = [data[i*batch_size: (i+1)*batch_size] for i in range(num_batches)]
        return data_list
    
    elif batch_mode == "minutes":
        window_length_ns = int(cfg.preprocessing.build_graphs.time_window_size*60_000_000_000)
        sliding_ns = int(batch_size*60_000_000_000) # min to ns
    
        t0 = data.t.min()
        t1 = data.t.max()
        t0_aligned = (t0 // sliding_ns) * sliding_ns

        # Mapping from window index to list of data points
        window_data = {}

        for p in data:
            # Compute window indices for each data point
            i0 = ((p.t - window_length_ns - t0_aligned) + sliding_ns - 1) // sliding_ns
            i1 = (p.t - t0_aligned) // sliding_ns
            i0 = max(i0, 0)  # Ensure i0 is non-negative

            for i in range(i0, i1 + 1):
                window_data.setdefault(i, []).append(p)

        # Build the list of windows
        windows = []
        for i in sorted(window_data.keys()):
            s = t0_aligned + i * sliding_ns          # Window start time (ns)
            e = s + window_length_ns                 # Window end time (ns)
            data_in_window = window_data[i]
            windows.append(collate_temporal_data(data_in_window))
        
        return windows
    
    raise ValueError(f"Invalid or missing batch mode {batch_mode}")

class _Cache:
    def __init__(self, shape, device):
        self._cache = torch.zeros(shape, device=device)
    
    @property
    def cache(self):
        return self._cache
        
    def detach(self):
        self._cache = self._cache.detach()
        
    def to(self, device):
        self._cache = self._cache.to(device)
        return self

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
        self.cache = {}
        self.is_warning = False

    def node_features_reshape(self, edge_index, x_src, x_dst, max_num_node=None, x_is_tuple=False):
        """
        Converts node features in shape (E, d) to a shape (N, d).
        Returns x as a tuple (x_src, x_dst).
        """
        if edge_index.min() != 0 and not self.is_warning:
            print(f"Warning: reshaping features with non-reindexed edge index leads to large cache stored in GPU memory.")
            self.is_warning = True
        
        max_num_node = max_num_node + 1 if max_num_node else edge_index.max() + 1
        feature_dim = x_src.size(1)
        
        if feature_dim not in self.cache or self.cache[feature_dim].cache.shape[0] <= max_num_node:
            self.cache[feature_dim] = _Cache((max_num_node, feature_dim), self.device)
        self.cache[feature_dim].detach()
        
        # To avoid storing gradients from all nodes, we detach() BEFORE caching. If we detach()
        # after storing, we loose the gradient for all operations happening before the reindexing.
        output = self.cache[feature_dim].cache
        output.detach()
        output.zero_()
        
        if x_is_tuple:
            scatter(x_src, edge_index[0], out=output, dim=0, reduce='mean')
            x_src_result = output.clone()
            output.zero_()
            
            scatter(x_dst, edge_index[1], out=output, dim=0, reduce='mean')
            x_dst_result = output.clone()
            return x_src_result[:max_num_node], x_dst_result[:max_num_node]
        else:
            scatter(x_src, edge_index[0], out=output, dim=0, reduce='mean')
            scatter(x_dst, edge_index[1], out=output, dim=0, reduce='mean')
            return output[:max_num_node]
    
    def reindex_graph(self, data, x_is_tuple=False):
        """
        Reindexes edge_index from 0 + reshapes node features.
        The original edge_index and node IDs are also kept.
        """
        data = data.clone()
        data.original_edge_index = data.edge_index
        x, data.edge_index, n_id = self._reindex_graph(data.edge_index, data.x_src, data.x_dst, x_is_tuple=x_is_tuple)
        if x_is_tuple:
            data.x_src, data.x_dst = x
        else:
            data.x = x
        
        data.original_n_id = n_id
        
        # When it's node-level detection, we also need to reshape the node types if we do node type pred. (e.g. ThreaTrace)
        if hasattr(data, "node_type_src"):
            data.node_type, *_ = self._reindex_graph(data.edge_index, data.node_type_src, data.node_type_dst, x_is_tuple=False)
        
        return data
    
    def _reindex_graph(self, edge_index, x_src, x_dst, x_is_tuple=False):
        """
        Reindexes edge_index with indices starting from 0.
        Also reshapes the node features.
        """
        if self.assoc is None:
            self.assoc = torch.empty((self.num_nodes, ), dtype=torch.long, device=self.device)

        n_id = edge_index.unique()
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.assoc.device)
        edge_index = self.assoc[edge_index]
        
        # Associates each feature vector to each reindexed node ID
        x = self.node_features_reshape(edge_index, x_src, x_dst, x_is_tuple=x_is_tuple)
        
        return x, edge_index, n_id
    
    def to(self, device):
        self.device = device
        if self.assoc is not None:
            self.assoc = self.assoc.to(device)

        for k, v in self.cache.items():
            self.cache[k] = v.to(device)
        return self
        

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
        torch.load(os.path.join(path, "state_dict.pkl")))
    
    if isinstance(model.encoder, TGNEncoder):
        model.encoder.neighbor_loader = torch.load(os.path.join(path, "neighbor_loader.pkl"))
        if cfg.detection.gnn_training.encoder.tgn.use_memory or "time_encoding" in cfg.detection.gnn_training.encoder.edge_features:
            model.encoder.memory = torch.load(os.path.join(path, "memory.pkl"))

    return model
