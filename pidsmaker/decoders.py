import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from pidsmaker.custom_mlp import CustomMLPAsbtract
from pidsmaker.utils.utils import compute_class_weights, log


# Basic blocks
class CustomMLPDecoder(CustomMLPAsbtract):
    def __init__(self, in_dim, out_dim, architecture, dropout):
        super().__init__(in_dim, out_dim, architecture, dropout)

    def forward(self, h):
        h = self.mlp(h)
        return h


class CustomEdgeMLP(CustomMLPAsbtract):
    def __init__(self, in_dim, out_dim, architecture, dropout, src_dst_projection_coef):
        super().__init__(in_dim * 2 * src_dst_projection_coef, out_dim, architecture, dropout)

        self.lin_src = nn.Linear(in_dim, in_dim * src_dst_projection_coef)
        self.lin_dst = nn.Linear(in_dim, in_dim * src_dst_projection_coef)

    def forward(self, h_src, h_dst):
        h = torch.cat([self.lin_src(h_src), self.lin_dst(h_dst)], dim=-1)
        h = self.mlp(h)
        return h


# Decoders
class NodLinkDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(NodLinkDecoder, self).__init__()

        h_dim = in_dim // 2
        self.encoder = self._NodLinkEncoder(in_dim, h_dim, device)
        self.decoder = self._NodeLinkDecoder(h_dim, out_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    class _NodLinkEncoder(nn.Module):
        def __init__(self, in_dim, h_dim, device):
            super().__init__()
            self.linear1 = nn.Linear(in_dim, h_dim)
            self.linear2 = nn.Linear(h_dim, h_dim // 2)
            self.linear3 = nn.Linear(h_dim // 2, h_dim // 4)
            self.linear4 = nn.Linear(h_dim // 2, h_dim // 4)

            self.N = torch.distributions.Normal(0, 1)
            self.N.loc = self.N.loc.to(device)
            self.N.scale = self.N.scale.to(device)
            self.kl = 0

        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            mu = self.linear3(x)
            sigma = torch.exp(self.linear4(x))
            z = mu + sigma * self.N.sample(mu.shape)
            self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
            return z

    class _NodeLinkDecoder(nn.Module):
        def __init__(self, h_dim, out_dim):
            super().__init__()
            self.linear1 = nn.Linear(h_dim // 4, h_dim // 2)
            self.linear2 = nn.Linear(h_dim // 2, h_dim)
            self.linear3 = nn.Linear(h_dim, out_dim)

        def forward(self, z):
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

        class_weights = (
            compute_class_weights(node_type, num_classes=self.node_type_dim)
            if self.balanced_loss
            else None
        )

        node_type_classes = node_type.argmax(dim=1)
        # TODO: IMPORTANT ERROR HERE, should be h_hat instead of h, but if I change
        # all results for systems using node type pred have to be re-run
        # NOTE: I fixed this error after Velox
        loss = self.loss_fn(h_hat, node_type_classes, inference=inference, weight=class_weights)
        return {"loss": loss, "out": F.log_softmax(h, dim=1)}


class EdgeEmbReconstruction(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(EdgeEmbReconstruction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, h_src, h_dst, inference, **kwargs):
        h_edge_hat = self.decoder(h_src=h_src, h_dst=h_dst)

        h_edge = torch.cat([h_src, h_dst], dim=-1)
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
        h = self.decoder(h_src=h_src, h_dst=h_dst)

        class_weights = (
            compute_class_weights(edge_type, num_classes=self.edge_type_dim)
            if self.balanced_loss
            else None
        )

        edge_type_classes = edge_type.argmax(dim=1)
        loss = self.loss_fn(h, edge_type_classes, inference=inference, class_weights=class_weights)
        return {"loss": loss}


class EdgeTypePredictionHetero(nn.Module):
    """Adds a projection head after the decoder, between each possible pair of entites"""

    def __init__(self, decoder, loss_fn, edge_type_predictors, ntype2edgemap):
        super().__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.edge_type_predictors = edge_type_predictors
        self.ntype2edgemap = ntype2edgemap

    def forward(self, h_src, h_dst, batch, inference, **kwargs):
        src_type_idx = batch.node_type_src_argmax
        dst_type_idx = batch.node_type_dst_argmax
        edge_type_classes = batch.edge_type_argmax

        h = self.decoder(h_src=h_src, h_dst=h_dst)

        losses = []
        for (src_type, dst_type), event_map in self.ntype2edgemap.items():
            mask = (src_type_idx == src_type) & (dst_type_idx == dst_type)
            if mask.any():
                layer = self.edge_type_predictors[f"{src_type}_{dst_type}"]
                out = layer(h[mask])

                reindexed_edge_types = event_map[edge_type_classes[mask]]
                loss = self.loss_fn(out, reindexed_edge_types, inference=inference)
                losses.append(loss)

        if inference:
            tot_loss = torch.cat(losses)
        else:
            tot_loss = torch.stack(losses).mean()
        return {"loss": tot_loss}


class FewShotEdgeDetection(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(FewShotEdgeDetection, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, h_src, h_dst, y_edge, inference, **kwargs):
        y_edge_hat = self.decoder(h_src=h_src, h_dst=h_dst)

        if inference:
            # To get an anomaly score instead of a loss, we get the complement of the softmax normalized score
            return {"loss": 1 - F.softmax(y_edge_hat, dim=1)[torch.arange(y_edge.shape[0]), y_edge]}

        return {
            "loss": self.loss_fn(y_edge_hat, y_edge, inference=inference)
        }  # weight=torch.tensor([1.0, 20000.0]).to(h_src.device))


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
        neg_edge_index = negative_sampling(
            edge_index, num_nodes=h.shape[0], num_neg_samples=pos_src.size(0)
        )
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

    def __init__(self, objective, graph_reindexer, is_edge_type_prediction, use_few_shot):
        super().__init__()
        self.objective = objective
        self.graph_reindexer = graph_reindexer
        self.is_edge_type_prediction = is_edge_type_prediction

        self.use_few_shot = use_few_shot
        self.losses = []
        self.ys = []

    def get_val_score(self) -> float:
        if self.use_few_shot:
            self.losses = torch.tensor(self.losses)
            self.ys = torch.tensor(self.ys)
            malicious_mean = self.losses[self.ys == 1].mean()
            benign_mean = self.losses[self.ys == 0].mean()
            log(f"=> Total mean score of fake malicious edges: {malicious_mean:.4f}")
            log(f"=> Total mean score of benign malicious edges: {benign_mean:.4f}")
            # ap = ap_score(self.ys, self.losses)
            score = malicious_mean - benign_mean
        else:
            score = 0.0

        self.losses = []
        self.ys = []
        return score

    def forward(self, edge_type, y_edge, validation, inference, *args, **kwargs):
        results = self.objective(
            *args, edge_type=edge_type, y_edge=y_edge, inference=inference, **kwargs
        )
        assert isinstance(results, dict), "Return value of an objective should be a dict"

        loss_or_losses = results["loss"]

        if validation and self.use_few_shot:
            self.losses.extend(loss_or_losses.cpu())
            self.ys.extend(y_edge.cpu())

        return results

    def _fast_negative_sampling(self, edge_index, h_src, h_dst, aug_coef=1.0, **kwargs):
        (h_src, h_dst), edge_index, n_id = self.graph_reindexer._reindex_graph(
            edge_index, h_src, h_dst, x_is_tuple=True
        )

        candidates = (
            edge_index.unique()
        )  # we consider both src and dst nodes in the batch as candidates to be negative
        neg_idx = torch.randint(0, len(candidates), (len(edge_index[0]),))
        candidates = candidates[neg_idx]

        neg_ei = torch.stack([edge_index[0], candidates])

        num_nodes = (edge_index.max() + 1).item()
        el_hash = lambda x: x[0, :] + x[1, :] * num_nodes
        el1d = el_hash(edge_index)

        neg_ei = neg_ei[:, neg_ei[0] != neg_ei[1]]  # remove self-loops
        neg_hash = el_hash(neg_ei)

        neg_samples = neg_ei[
            :, ~torch.isin(neg_hash, el1d)
        ]  # remove collision edges in positive edges

        neg_h_src = h_src[neg_samples[0]]
        neg_h_dst = h_dst[neg_samples[1]]

        return neg_h_src, neg_h_dst


class EdgeContrastiveDecoder(nn.Module):
    def __init__(self, decoder, loss_fn, graph_reindexer):
        super().__init__()

        self.decoder = decoder
        self.loss_fn = loss_fn
        self.graph_reindexer = graph_reindexer

    def _fast_negative_sampling(self, edge_index, h_src, h_dst, aug_coef=1.0):
        (h_src, h_dst), edge_index, _ = self.graph_reindexer._reindex_graph(
            edge_index, h_src, h_dst, x_is_tuple=True
        )

        neg_dst_idx = torch.randperm(
            len(edge_index[0])
        )  # TODO: we may want to try also putting source nodes in the neg dst nodes
        neg_dst = edge_index[1, neg_dst_idx]
        neg_ei = torch.stack([edge_index[0], neg_dst])

        num_nodes = (edge_index.max() + 1).item()
        el_hash = lambda x: x[0, :] + x[1, :] * num_nodes
        el1d = el_hash(edge_index)

        neg_ei = neg_ei[:, neg_ei[0] != neg_ei[1]]  # remove self-loops
        neg_hash = el_hash(neg_ei)

        neg_samples = neg_ei[
            :, ~torch.isin(neg_hash, el1d)
        ]  # remove collision edges in positive edges

        neg_h_src = h_src[neg_samples[0]]
        neg_h_dst = h_dst[neg_samples[1]]

        return neg_h_src, neg_h_dst

    def forward(self, h_src, h_dst, edge_index, inference, **kwargs):
        pos_scores = self.decoder(h_src=h_src, h_dst=h_dst)

        if not inference:
            neg_h_src, neg_h_dst = self._fast_negative_sampling(edge_index, h_src, h_dst)
            neg_scores = self.decoder(h_src=neg_h_src, h_dst=neg_h_dst)
        else:
            neg_scores = None

        # class_weights = compute_class_weights(node_type, num_classes=self.node_type_dim) \
        # if self.balanced_loss else None

        loss = self.loss_fn(pos_scores, neg_scores, inference=inference)
        return {"loss": loss.squeeze()}


class EdgeLinearDecoder(nn.Module):
    def __init__(self, in_dim, dropout):
        super().__init__()

        self.lin_src = nn.Linear(in_dim, in_dim)
        self.lin_dst = nn.Linear(in_dim, in_dim)
        self.lin_final = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_src, h_dst):
        h = self.lin_src(self.drop(h_src)) + self.lin_dst(self.drop(h_dst))
        h = h.relu()
        return self.lin_final(h)


class EdgeInnerProductDecoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.drop = nn.Dropout(dropout)

    def forward(self, h_src, h_dst):
        return (self.drop(h_src) * self.drop(h_dst)).sum(dim=1)
