import torch
from torch_geometric.data import Data
from torch_geometric.loader import TemporalDataLoader


def custom_temporal_data_loader(data, batch_size, *args, **kwargs):
    """
    A simple `TemporalDataLoader` which also update the edge_index with the
    sampled edges of size `batch_size`. By default, only attributes of shape (E, d)
    are updated, `edge_index` is thus not updated automatically.
    """
    loader = TemporalDataLoader(data, batch_size=batch_size, *args, **kwargs)
    for batch in loader:
        batch.edge_index = torch.stack([batch.src, batch.dst])
        yield batch

def temporal_data_to_data(data):
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
        self.assoc = None
        self.num_nodes = num_nodes
        self.device = device
            
    def __call__(self, data):
        if self.assoc is None:
            self.assoc = torch.empty((self.num_nodes, ), dtype=torch.long, device=self.device)
            
        (data.x_src, data.x_dst), data.edge_index = self._reindex_graph(data.edge_index, data.x_src, data.x_dst)
        return data

    def _reindex_graph(self, edge_index, batch_x_src, batch_x_dst):
        # Reindex edge_index with indices starting from 0
        n_id = edge_index.unique()
        self.assoc[n_id] = torch.arange(n_id.size(0), device=edge_index.device)
        edge_index = self.assoc[edge_index]
        
        # Associates each feature vector to each reindexed node ID
        x_src = torch.zeros((edge_index.max() + 1, batch_x_src.shape[1]), device=edge_index.device)
        x_dst = x_src.clone()
        
        x_src[edge_index[0, :]] = batch_x_src
        x_dst[edge_index[1, :]] = batch_x_dst
        x = (x_src, x_dst)
        
        return x, edge_index
