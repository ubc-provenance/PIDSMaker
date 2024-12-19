from provnet_utils import *
from config import *
import torch.nn as nn
from encoders import TGNEncoder
from experiments.uncertainty import activate_dropout_inference
from decoders import NodeTypePrediction


class Model(nn.Module):
    def __init__(self,
            encoder: nn.Module,
            decoders: list[nn.Module],
            num_nodes: int,
            in_dim: int,
            out_dim: int,
            use_contrastive_learning: bool,
            device,
            graph_reindexer,
            node_level,
            is_running_mc_dropout,
        ):
        super(Model, self).__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.use_contrastive_learning = use_contrastive_learning
        self.graph_reindexer = graph_reindexer
        self.device = device
        
        self.last_h_storage, self.last_h_non_empty_nodes = None, None
        if self.use_contrastive_learning:
            self.last_h_storage = torch.empty((num_nodes, out_dim), device=device)
            self.last_h_non_empty_nodes = torch.tensor([], dtype=torch.long, device=device)
            
        self.node_level = node_level
        self.is_running_mc_dropout = is_running_mc_dropout
        
    def embed(self, batch, full_data, inference=False, **kwargs):
        train_mode = not inference
        x = self._reshape_x(batch)
        edge_index = batch.edge_index
        with torch.set_grad_enabled(train_mode):
            h = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x=x,
                msg=batch.msg,
                edge_feats=getattr(batch, "edge_feats", None),
                full_data=full_data, # NOTE: warning, this object contains the full graph without TGN sampling
                inference=inference,
                edge_types= batch.edge_type
            )
        return h
        
    def forward(self, batch, full_data, inference=False, validation=False):
        train_mode = not inference
        x = self._reshape_x(batch)
        edge_index = batch.edge_index
        num_nodes = len(batch.edge_index.unique())

        with torch.set_grad_enabled(train_mode):
            h = self.embed(batch, full_data, inference=inference)
            
            if isinstance(h, tuple):
                h_src, h_dst = h
            else:
                # TGN encoder returns a pair h_src, h_dst whereas other encoders return simply h as shape (N, d)
                # Here we simply transform to get a shape (E, d)
                h_src, h_dst = (h[edge_index[0]], h[edge_index[1]]) \
                    if isinstance(h, torch.Tensor) else h
            
            if self.node_level:
                if isinstance(h, tuple):
                    h, _, n_id = self.graph_reindexer._reindex_graph(edge_index, h[0], h[1]) # TODO: duplicate with the one in TGN encoder, remove
                    batch.original_n_id = n_id
                if isinstance(x, tuple):
                    x, _, n_id = self.graph_reindexer._reindex_graph(edge_index, x[0], x[1])
                    batch.original_n_id = n_id
                
            else:
                if not isinstance(x, tuple):
                    x = (batch.x_src[edge_index[0]], batch.x_dst[edge_index[1]])
            
            # Train mode: loss | Inference mode: scores
            loss_or_scores = None
            
            for objective in self.decoders:
                node_type = self.get_node_type_if_needed(batch, objective)
                    
                results = objective(
                    h_src=h_src, # shape (E, d)
                    h_dst=h_dst, # shape (E, d)
                    h=h, # shape (N, d)
                    x=x,
                    edge_index=edge_index,
                    edge_type=batch.edge_type,
                    inference=inference,
                    last_h_storage=self.last_h_storage,
                    last_h_non_empty_nodes=self.last_h_non_empty_nodes,
                    node_type=node_type,
                    validation=validation,
                )
                loss = results["loss"]
                
                if loss_or_scores is None:
                    loss_or_scores = (torch.zeros(1) if train_mode else \
                        torch.zeros(loss.shape[0], dtype=torch.float)).to(edge_index.device)
                
                if loss.numel() != loss_or_scores.numel():
                    raise TypeError(f"Shapes of loss/score do not match ({loss.numel()} vs {loss_or_scores.numel()})")
                loss_or_scores = loss_or_scores + loss

            return results
    
    def _reshape_x(self, batch):
        if self.node_level and hasattr(batch, "x"):
            x = batch.x
        else:
            x = (batch.x_src, batch.x_dst)
        return x
        
    def get_val_ap(self):
        # If multiple decoders are used, we take the average of the val scores
        return np.mean([d.get_val_score() for d in self.decoders])

    def to_device(self, device):
        if self.device == device:
            return self
        
        for decoder in self.decoders:
            decoder.graph_reindexer.to(device)
        
        if isinstance(self.encoder, TGNEncoder):
            self.encoder.to_device(device)
            
        self.device = device
        self.graph_reindexer.to(device)
        return self.to(device)

    # override
    def eval(self):
        super().eval()
        
        if self.is_running_mc_dropout:
            activate_dropout_inference(self)

    def get_node_type_if_needed(self, batch, objective):
        node_type = getattr(batch, "node_type", None)
        # Special case when TGN is used with node type pred, the batch is not already reindexed so we reindex
        # only to get node types as shape (N, d)
        if isinstance(objective.objective, NodeTypePrediction) and node_type is None:
            reindexed_batch = self.graph_reindexer.reindex_graph(batch)
            node_type = reindexed_batch.node_type
        return node_type
    