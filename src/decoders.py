from provnet_utils import *
from config import *
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from sklearn.metrics import average_precision_score as ap_score


# Decoders
class EdgeMLPDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, architecture):
        super(EdgeMLPDecoder, self).__init__()
        
        self.lin_src = Linear(in_dim, in_dim*2) # TODO: not sure it's good to project in *2, try *1 instead
        self.lin_dst = Linear(in_dim, in_dim*2)
        
        self.mlp = build_mlp_from_string(architecture, in_dim * 4, out_dim)

    def forward(self, h_src, h_dst):
        h = torch.cat([self.lin_src(h_src), self.lin_dst(h_dst)], dim=-1)
        h = self.mlp(h)
        return h
    
class NodeMLPDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, architecture):
        super(NodeMLPDecoder, self).__init__()
        
        self.mlp = build_mlp_from_string(architecture, in_dim, out_dim)

    def forward(self, h):
        h = self.mlp(h)
        return h

class NodLinkDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NodLinkDecoder, self).__init__()
        
        h_dim = in_dim // 2
        self.encoder = self._NodLinkEncoder(in_dim, h_dim)
        self.decoder = self._NodeLinkDecoder(h_dim, out_dim)

    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)
    
    class _NodLinkEncoder(nn.Module):
        def __init__(self, in_dim, h_dim):
            super().__init__()
            self.linear1 = nn.Linear(in_dim, h_dim)
            self.linear2 = nn.Linear(h_dim, h_dim // 2)
            self.linear3 = nn.Linear(h_dim // 2, h_dim // 4)
            self.linear4 = nn.Linear(h_dim // 2, h_dim // 4)

            self.N = torch.distributions.Normal(0,1)
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
            self.kl = 0

        def forward(self,x):
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            mu = self.linear3(x)
            sigma = torch.exp(self.linear4(x))
            z = mu + sigma*self.N.sample(mu.shape)
            self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            return z

    class _NodeLinkDecoder(nn.Module):
        def __init__(self, h_dim, out_dim):
            super().__init__()
            self.linear1 = nn.Linear(h_dim // 4, h_dim // 2)
            self.linear2 = nn.Linear(h_dim // 2, h_dim)
            self.linear3 = nn.Linear(h_dim, out_dim)

        def forward(self,z):
            z = F.relu(self.linear1(z))
            z = F.relu(self.linear2(z))
            z = self.linear3(z)
            return z

# Objectives
class NodeFeatReconstruction(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(NodeFeatReconstruction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, h, x, inference, **kwargs):
        x_hat = self.decoder(h)
        loss = self.loss_fn(x_hat, x, inference=inference)
        return {"loss": loss}
    
class NodeEmbReconstruction(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(NodeEmbReconstruction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, h, inference, **kwargs):
        h_hat = self.decoder(h)
        loss = self.loss_fn(h_hat, h, inference=inference)
        return {"loss": loss}

class NodeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, node_type_dim):
        super(NodeTypePrediction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.node_type_dim = node_type_dim
        
    def forward(self, h, node_type, inference, **kwargs):
        h_hat = self.decoder(h)
        
        class_weights = compute_class_weights(node_type, num_classes=self.node_type_dim) \
            if self.balanced_loss else None

        node_type_classes = node_type.argmax(dim=1)
        # TODO: IMPORTANT ERROR HERE, should be h_hat instead of h, but if I change
        # all results for systems using node type pred have to be re-run
        loss = self.loss_fn(h, node_type_classes, inference=inference, weight=class_weights)
        return {"loss": loss, "out": F.log_softmax(h, dim=1)}

class EdgeEmbReconstruction(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(EdgeEmbReconstruction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, h_src, h_dst, inference, **kwargs):
        h_edge = torch.cat([h_src, h_dst], dim=-1)
        h_edge_hat = self.decoder(h_edge)
        
        loss = self.loss_fn(h_edge_hat, h_edge, inference=inference)
        return {"loss": loss}

class EdgeTypePrediction(nn.Module):
    def __init__(self, decoder, loss_fn, balanced_loss, edge_type_dim):
        super(EdgeTypePrediction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.balanced_loss = balanced_loss
        self.edge_type_dim = edge_type_dim
        
    def forward(self, h_src, h_dst, edge_type, inference, **kwargs):
        h = self.decoder(h_src, h_dst)
        
        class_weights = compute_class_weights(edge_type, num_classes=self.edge_type_dim) \
            if self.balanced_loss else None
        
        edge_type_classes = edge_type.argmax(dim=1)
        loss = self.loss_fn(h, edge_type_classes, inference=inference, class_weights=class_weights)
        return {"loss": loss}

class GMAEFeatReconstruction(nn.Module):
    def __init__(self, decoder, loss_fn, mask_rate):
        super(GMAEFeatReconstruction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.mask_rate = mask_rate

    def generate_mask_token(self, x):
        # generate mask_token dynamically
        mask_token = torch.zeros_like(x[0])
        return mask_token

    def forward(self, x, h, edge_index, inference, **kwargs):
        mask_rate = self.mask_rate
        num_nodes = x.shape[0]
        num_mask_nodes = int(mask_rate * num_nodes)
        perm = torch.randperm(num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        mask_token = self.generate_mask_token(x)

        # TODO: Nothing is masked here actually, and mask_token and x_masked are not even used
        # We simply compute the loss on a subset of nodes but there is no masking
        x_masked = x.clone()
        x_masked[mask_nodes] = mask_token.to(x.device)

        recon = self.decoder(h, edge_index)

        x_init = x[mask_nodes].to(h.device)
        x_rec = recon[mask_nodes].to(h.device)
        
        # TODO: During inference, we should return for each node  its own loss
        # Now, with the current code it returns only the loss for masked nodes so
        # We can use this loss in the final loss because the shape mismatch.
        # For now, I simply return 0 to avoid error, so that there is only the other GMAEStructPrediction considered
        if inference:
            losses = torch.zeros((x.shape[0],), device=x.device)
            return {"loss": losses}
            
        loss = self.loss_fn(x_rec, x_init, inference=inference)

        return {"loss": loss}

class GMAEStructPrediction(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(GMAEStructPrediction, self).__init__()
        self.edge_recon_fc = decoder
        self.loss_fn = loss_fn

    def forward(self, h, edge_index, inference, **kwargs):

        pos_src, pos_dst = edge_index
        neg_edge_index = negative_sampling(edge_index, num_nodes=h.shape[0], num_neg_samples=pos_src.size(0))
        h_src, h_dst = h[edge_index[0]], h[edge_index[1]]

        pos_samples = torch.cat([h_src, h_dst], dim=-1)
        neg_samples = torch.cat([h[neg_edge_index[0]], h[neg_edge_index[1]]], dim=-1)

        y_pred_pos = self.edge_recon_fc(pos_samples).squeeze(-1)
        y_pred_neg = self.edge_recon_fc(neg_samples).squeeze(-1)

        y_true = torch.cat([torch.ones_like(y_pred_pos), torch.zeros_like(y_pred_neg)])
        y_pred = torch.cat([y_pred_pos, y_pred_neg])
        
        # TODO: Here it returns a loss for each edge so we can't use it directly at node level
        # and combine it with node level
        if inference:
            losses = torch.zeros((h.shape[0],), device=h.device)
            return {"loss": losses}
        
        loss = self.loss_fn(y_pred, y_true, inference=inference)
        
        return {"loss": loss}

class ValidationContrastiveStopper(nn.Module):
    """
    Binded to a an objective object to store the scores seen in the validation set
    and compute a score to pick the best model without reliying on data snooping
    and test set.
    """
    def __init__(self, objective, graph_reindexer, is_edge_type_prediction):
        super().__init__()
        self.objective = objective
        self.graph_reindexer = graph_reindexer
        self.is_edge_type_prediction = is_edge_type_prediction
        
        self.positives = []
        self.negatives = []
    
    def get_val_score(self) -> float:
        if len(self.positives) == 0:
            raise ValueError(f"The forward function has likely not been called.")

        # Contrastive val AP loss
        # if self.is_edge_type_prediction:
            # negatives = torch.tensor(self.negatives)
            # positives = torch.tensor(self.positives)
            # labels = torch.cat([torch.ones_like(negatives), torch.zeros_like(positives)])
            # scores = torch.cat([negatives, positives])
            # ap = ap_score(labels, scores)
            
            # self.positives = []
            # self.negatives = []
            # return ap
        
        # For other objectives, we simply want to pick the model with the max
        # loss on the validation set, so we take the complement and we scale between 0-1 to match AP's range
        score = 1 - torch.tensor(self.positives).mean().item()
        self.positives = []
        return score
    
    def forward(self, edge_type, validation, inference, *args, **kwargs):
        results = self.objective(*args, edge_type=edge_type, inference=inference, **kwargs)
        assert isinstance(results, dict), "Return value of an objective should be a dict"
        
        loss_or_losses = results["loss"]
        
        if validation:
            # We only support contrastive val AP loss for edge type prediction
            if self.is_edge_type_prediction:
                neg_h_src, neg_h_dst = self._fast_negative_sampling(**kwargs)
                edge_type = edge_type[:len(neg_h_src)]
                neg_loss_or_losses = self.objective(h_src=neg_h_src, h_dst=neg_h_dst, edge_type=edge_type, inference=inference)["loss"]
                
                self.positives.extend(loss_or_losses)
                self.negatives.extend(neg_loss_or_losses)
                
            else:
                self.positives.extend(loss_or_losses)
        
        return results
        
    def _fast_negative_sampling(self, edge_index, h_src, h_dst, aug_coef=1.0, **kwargs):
        (h_src, h_dst), edge_index, n_id = self.graph_reindexer._reindex_graph(edge_index, h_src, h_dst, x_is_tuple=True)
        
        candidates = edge_index.unique() # we consider both src and dst nodes in the batch as candidates to be negative
        neg_idx = torch.randint(0, len(candidates), (len(edge_index[0]),))
        candidates = candidates[neg_idx]

        neg_ei = torch.stack([edge_index[0], candidates])
        
        num_nodes = (edge_index.max() + 1).item()
        el_hash = lambda x: x[0, :] + x[1, :] * num_nodes
        el1d = el_hash(edge_index)
        
        neg_ei = neg_ei[:, neg_ei[0] != neg_ei[1]]  # remove self-loops
        neg_hash = el_hash(neg_ei)
        
        neg_samples = neg_ei[:, ~torch.isin(neg_hash, el1d)]  # remove collision edges in positive edges
        
        neg_h_src = h_src[neg_samples[0]]
        neg_h_dst = h_dst[neg_samples[1]]
        
        return neg_h_src, neg_h_dst
        

# class EdgeContrastiveDecoder(nn.Module):
#     def __init__(self, decoder, loss_fn, graph_reindexer, neg_sampling_method):
#         super().__init__()
        
#         self.decoder = decoder
#         self.loss_fn = loss_fn
#         self.graph_reindexer = graph_reindexer
#         self.neg_sampling_method = neg_sampling_method
    
#     def _fast_negative_sampling(self, edge_index, h_src, h_dst, aug_coef=1.0):
#         (h_src, h_dst), edge_index = self.graph_reindexer._reindex_graph(edge_index, h_src, h_dst)
        
#         neg_dst_idx = torch.randperm(len(edge_index[0]))  # TODO: we may want to try also putting source nodes in the neg dst nodes
#         neg_dst = edge_index[1, neg_dst_idx]
#         neg_ei = torch.stack([edge_index[0], neg_dst])
        
#         num_nodes = (edge_index.max() + 1).item()
#         el_hash = lambda x: x[0, :] + x[1, :] * num_nodes
#         el1d = el_hash(edge_index)
        
#         neg_ei = neg_ei[:, neg_ei[0] != neg_ei[1]]  # remove self-loops
#         neg_hash = el_hash(neg_ei)
        
#         neg_samples = neg_ei[:, ~torch.isin(neg_hash, el1d)]  # remove collision edges in positive edges
        
#         neg_h_src = h_src[neg_samples[0]]
#         neg_h_dst = h_dst[neg_samples[1]]
        
#         return neg_h_src, neg_h_dst

#     def forward(self, h_src, h_dst, edge_index, last_h_storage, last_h_non_empty_nodes, inference, **kwargs):
#         pos_scores = self.decoder(h_src=h_src, h_dst=h_dst)
        
#         if not inference:
#             if self.neg_sampling_method == "nodes_in_current_batch":
#                 neg_h_src, neg_h_dst = self._fast_negative_sampling(edge_index, h_src, h_dst)
#                 neg_scores = self.decoder(h_src=neg_h_src, h_dst=neg_h_dst)
            
#             elif self.neg_sampling_method == "previously_seen_nodes":
#                 neg_dst_candidates = last_h_non_empty_nodes[~torch.isin(last_h_non_empty_nodes, edge_index[1].unique())]
#                 neg_dst = torch.randperm(neg_dst_candidates.numel())[:edge_index.shape[1]]
#                 neg_scores = self.decoder(h_src=h_src[:neg_dst.numel()], h_dst=last_h_storage[neg_dst])
        
#         else:
#             neg_scores = None
        
#         loss = self.loss_fn(pos_scores, neg_scores, inference=inference)
#         return loss.squeeze()

# class EdgeLinearDecoder(nn.Module):
#     def __init__(self, in_dim, dropout):
#         super().__init__()
        
#         self.lin_src = Linear(in_dim, in_dim)
#         self.lin_dst = Linear(in_dim, in_dim)
#         self.lin_final = Linear(in_dim, 1)
#         self.drop = nn.Dropout(dropout)

#     def forward(self, h_src, h_dst):
#         h = self.lin_src(self.drop(h_src)) + self.lin_dst(self.drop(h_dst))
#         h = h.relu()
#         return self.lin_final(h)

# class EdgeInnerProductDecoder(nn.Module):
#     def __init__(self, dropout):
#         super().__init__()

#         self.drop = nn.Dropout(dropout)

#     def forward(self, h_src, h_dst):
#         return (self.drop(h_src) * self.drop(h_dst)).sum(dim=1)
