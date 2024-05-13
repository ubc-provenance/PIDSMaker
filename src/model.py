from provnet_utils import *
from config import *
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,
            encoder: nn.Module,
            decoders: list[nn.Module],
            num_nodes: int,
            in_dim: int,
            out_dim: int,
            use_contrastive_learning: bool,
            device,
            use_tgn: bool,
        ):
        super(Model, self).__init__()

        self.encoder = encoder
        self.decoders = decoders
        self.use_contrastive_learning = use_contrastive_learning
        
        self.last_h_storage, self.last_h_non_empty_nodes = None, None
        if self.use_contrastive_learning:
            self.last_h_storage = torch.empty((num_nodes, out_dim), device=device)
            self.last_h_non_empty_nodes = torch.tensor([], dtype=torch.long, device=device)
            
        self.use_tgn = use_tgn
        if not use_tgn:
            self.assoc = torch.empty((num_nodes, ), dtype=torch.long, device=device)
        
    def forward(self, batch, full_data, inference=False):
        train_mode = not inference
        
        x = (batch.x_src, batch.x_dst)
        edge_index = torch.stack([batch.src, batch.dst])
        if not self.use_tgn:
            x, edge_index = self._relabel_graph(edge_index, batch.x_src, batch.x_dst)
        
        with torch.set_grad_enabled(train_mode):
            h_src, h_dst = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x=x,
                msg=batch.msg,
                edge_feats=batch.edge_feats,
                full_data=full_data, # NOTE: warning, this object contains the full graph without TGN sampling
                inference=inference,
            )
            if self.use_contrastive_learning:
                involved_nodes = torch.cat([batch.src, batch.dst])
                self.last_h_storage[involved_nodes] = torch.cat([h_src, h_dst]).detach()
                self.last_h_non_empty_nodes = torch.cat([involved_nodes, self.last_h_non_empty_nodes]).unique()
            
            # Train mode: loss | Inference mode: edge scores
            loss_or_scores = (torch.zeros(1) if train_mode else \
                torch.zeros(edge_index.shape[1], dtype=torch.float)).to(h_src.device)

            for decoder in self.decoders:
                loss = decoder(
                    h_src=h_src,
                    h_dst=h_dst,
                    edge_index=edge_index,
                    edge_type=batch.edge_type,
                    inference=inference,
                    last_h_storage=self.last_h_storage,
                    last_h_non_empty_nodes=self.last_h_non_empty_nodes,
                )
                loss_or_scores = loss_or_scores + loss
                
            return loss_or_scores

    def _relabel_graph(self, edge_index, batch_x_src, batch_x_dst):
        """
        Simply transforms an edge_index from a time window and its src/dst node features of shape (E, d)
        to an relabelled edge_index with node IDs starting from 0 and src/dst node features of shape
        (max_num_node + 1, d).
        This relabelling is essential for the graph to be computed by a standard GNN model with PyG.
        """
        
        # Relabel edge_index with indices starting from 0
        n_id = edge_index.unique()
        self.assoc[n_id] = torch.arange(n_id.size(0), device=edge_index.device)
        edge_index = self.assoc[edge_index]
        
        # Associates each feature vector to each relabeled node ID
        x_src = torch.zeros((edge_index.max() + 1, batch_x_src.shape[1]), device=edge_index.device)
        x_dst = x_src.clone()
        
        x_src[edge_index[0, :]] = batch_x_src
        x_dst[edge_index[1, :]] = batch_x_dst
        x = (x_src, x_dst)
        
        return x, edge_index
