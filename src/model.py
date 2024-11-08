from provnet_utils import *
from config import *
import torch.nn as nn
from encoders import TGNEncoder
from experiments.uncertainty import activate_dropout_inference
from sklearn.metrics import average_precision_score as ap_score


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
            val_stopping_aug_coef,
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
        self.val_stopping_aug_coef = val_stopping_aug_coef
        
        self.positives = []
        self.negatives = []
        
    def embed(self, batch, full_data, inference=False, **kwargs):
        train_mode = not inference
        if self.node_level:
            x = batch.x
        else:
            x = (batch.x_src, batch.x_dst)
        edge_index = batch.edge_index
        with torch.set_grad_enabled(train_mode):
            h = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x=x,
                msg=batch.msg,
                edge_feats=batch.edge_feats if hasattr(batch, "edge_feats") else None,
                full_data=full_data, # NOTE: warning, this object contains the full graph without TGN sampling
                inference=inference,
                edge_types= batch.edge_type
            )
        return h
    
    def forward(self, batch, full_data, inference=False, validation=False):
        results = self._forward(batch, full_data, inference=inference, validation=validation)
        loss_or_losses = results["loss"]
        
        if validation:
            neg_batch, picked_idx = self._fast_negative_sampling(batch)
            neg_losses = self._forward(neg_batch, full_data, inference=inference, validation=validation, ignore_tgn=True)["loss"]
            
            if self.node_level:
                picked_idx = neg_batch.dst[picked_idx].unique()
                
            picked_neg_losses = neg_losses[picked_idx]
            picked_pos_losses = loss_or_losses[picked_idx]
            
            self.positives.extend(picked_pos_losses)
            self.negatives.extend(picked_neg_losses)
    
        return results
    
    # def _fast_negative_sampling(self, batch, **kwargs):
    #     edge_index = batch.edge_index
    #     num_edges = edge_index.size(1)
    #     val_stopping_aug_coef = self.val_stopping_aug_coef

    #     # Calculate the number of edges to modify based on val_stopping_aug_coef
    #     num_to_pick = int(num_edges * val_stopping_aug_coef)

    #     # Deterministically select edge indices to modify
    #     if num_to_pick > 0:
    #         step = max(num_edges // num_to_pick, 1)
    #         picked_idx = torch.arange(0, num_edges, step)[:num_to_pick]
    #     else:
    #         picked_idx = torch.tensor([], dtype=torch.long)

    #     if len(picked_idx) > 0:
    #         # Generate synthetic attack edges
    #         # Get the necessary attributes from the batch
            
    #         node_id_flat = torch.cat([batch.src, batch.dst])
    #         node_type_flat = torch.cat([batch.node_type_src, batch.node_type_dst])
    #         node_feat_flat = torch.cat([batch.x_src, batch.x_dst])

    #         subject_nodes = (node_type_flat[:,0] == 1).nonzero(as_tuple=True)[0].unique()
    #         file_nodes = (node_type_flat[:,1] == 1).nonzero(as_tuple=True)[0].unique()
    #         netflow_nodes = (node_type_flat[:,2] == 1).nonzero(as_tuple=True)[0].unique()

    #         # Ensure we have enough nodes to create attack patterns
    #         if len(subject_nodes) == 0 or (len(file_nodes) == 0 and len(netflow_nodes) == 0):
    #             raise ValueError("Not enough nodes to create attack patterns.")

    #         edge_type2one_hot = gen_relation_onehot(rel2id=rel2id_darpa_tc) # TODO: replace

    #         for idx in picked_idx:
    #             # Select a random attack pattern
    #             # For deterministic behavior, we'll cycle through predefined patterns
    #             pattern_type = idx % 3  # Three types of attack patterns

    #             if pattern_type == 0 and len(netflow_nodes) > 0:
    #                 # Pattern: subject -> netflow (EVENT_CONNECT)
    #                 src_node = subject_nodes[idx % len(subject_nodes)]
    #                 dst_node = netflow_nodes[idx % len(netflow_nodes)]
    #                 edge_type = edge_type2one_hot['EVENT_CONNECT']
    #             elif pattern_type == 1 and len(file_nodes) > 0:
    #                 # Pattern: subject -> file (EVENT_READ)
    #                 src_node = subject_nodes[idx % len(subject_nodes)]
    #                 dst_node = file_nodes[idx % len(file_nodes)]
    #                 edge_type = edge_type2one_hot['EVENT_READ']
    #             elif pattern_type == 2 and len(subject_nodes) > 1:
    #                 # Pattern: subject -> subject (EVENT_EXECUTE)
    #                 src_node = subject_nodes[idx % len(subject_nodes)]
    #                 dst_node = subject_nodes[(idx + 1) % len(subject_nodes)]
    #                 edge_type = edge_type2one_hot['EVENT_EXECUTE']
    #             else:
    #                 # Default pattern if specific nodes are unavailable
    #                 src_node = subject_nodes[idx % len(subject_nodes)]
    #                 dst_node = subject_nodes[(idx + 1) % len(subject_nodes)]
    #                 edge_type = edge_type2one_hot['EVENT_EXECUTE']
                
    #             # Replace the edge in edge_index
    #             batch.src[idx] = src_node
    #             batch.dst[idx] = dst_node
    #             batch.edge_type[idx] = edge_type

    #     return batch, picked_idx
    
    def _fast_negative_sampling(self, batch, **kwargs):
        edge_index = batch.edge_index
        num_edges = edge_index.size(1)
        val_stopping_aug_coef = self.val_stopping_aug_coef

        # Get unique nodes from edge_index as candidates for negative sampling
        candidates = edge_index.unique()
        num_candidates = len(candidates)

        # Deterministically select candidate nodes by repeating the candidates list if necessary
        neg_idx = torch.arange(num_edges) % num_candidates
        candidates_neg = candidates[neg_idx]

        # Calculate the number of edges to modify based on val_stopping_aug_coef
        num_to_pick = int(num_edges * val_stopping_aug_coef)

        # Deterministically select edge indices to modify
        if num_to_pick > 0:
            step = max(num_edges // num_to_pick, 1)
            picked_idx = torch.arange(0, num_edges, step)[:num_to_pick]
        else:
            picked_idx = torch.tensor([], dtype=torch.long)

        # Replace the destination nodes of the selected edges with the candidate nodes
        edge_index[1, picked_idx] = candidates_neg[picked_idx]

        batch.src = edge_index[0]
        batch.dst = edge_index[1]

        return batch, picked_idx
    
    def _forward(self, batch, full_data, inference=False, validation=False, ignore_tgn=False):
        train_mode = not inference
        if self.node_level:
            x = batch.x
        else:
            x = (batch.x_src, batch.x_dst)
        edge_index = batch.edge_index

        with torch.set_grad_enabled(train_mode):
            h = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x=x,
                msg=batch.msg,
                edge_feats=batch.edge_feats if hasattr(batch, "edge_feats") else None,
                full_data=full_data, # NOTE: warning, this object contains the full graph without TGN sampling
                inference=inference,
                edge_types= batch.edge_type,
                ignore_tgn=ignore_tgn,
            )

            num_elements = None
            if self.node_level:
                h_src, h_dst = None, None
                num_elements = h.shape[0]
            else:
                # TGN encoder returns a pair h_src, h_dst whereas other encoders return simply h as shape (N, d)
                # Here we simply transform to get a shape (E, d)
                h_src, h_dst = (h[edge_index[0]], h[edge_index[1]]) \
                    if isinstance(h, torch.Tensor) else h
            
                # Same for features
                if x[0].shape[0] != edge_index.shape[1]:
                    x = (batch.x_src[edge_index[0]], batch.x_dst[edge_index[1]])
                    
                num_elements = h_src.shape[0]

            # if self.use_contrastive_learning:
            #     involved_nodes = edge_index.flatten()
            #     self.last_h_storage[involved_nodes] = torch.cat([h_src, h_dst]).detach()
            #     self.last_h_non_empty_nodes = torch.cat([involved_nodes, self.last_h_non_empty_nodes]).unique()
            
            # Train mode: loss | Inference mode: scores
            loss_or_scores = (torch.zeros(1) if train_mode else \
                torch.zeros(num_elements, dtype=torch.float)).to(edge_index.device)
            
            for objective in self.decoders:
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
                    node_type=batch.node_type if hasattr(batch, "node_type") else None,
                    validation=validation,
                )
                loss = results["loss"]
                if loss.numel() != loss_or_scores.numel():
                    raise TypeError(f"Shapes of loss/score do not match ({loss.numel()} vs {loss_or_scores.numel()})")
                loss_or_scores = loss_or_scores + loss

            return results
        
    def get_val_ap(self):
        negatives = torch.tensor(self.negatives)
        positives = torch.tensor(self.positives)
        labels = torch.cat([torch.ones_like(negatives), torch.zeros_like(positives)])
        scores = torch.cat([negatives, positives])
        ap = ap_score(labels, scores)
        
        self.positives = []
        self.negatives = []
        return ap

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
