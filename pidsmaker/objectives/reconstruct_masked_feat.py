import torch
import torch.nn as nn


class GMAEFeatReconstruction(nn.Module):
    def __init__(self, decoder, loss_fn, mask_rate, embed_dim):
        super(GMAEFeatReconstruction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.mask_rate = mask_rate
        # Learnable mask token initialized to zeros, matching MAGIC-main
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def mask_input(self, x):
        """
        Masks the input features BEFORE encoding.
        Returns:
            x_masked: The input features with masked nodes replaced by token
            mask_nodes: Indices of masked nodes
        """
        num_nodes = x.shape[0]
        num_mask_nodes = int(self.mask_rate * num_nodes)
        perm = torch.randperm(num_nodes, device=x.device)
        mask_nodes = perm[:num_mask_nodes]

        x_masked = x.clone()
        x_masked[mask_nodes] = self.mask_token.to(x.device)

        return x_masked, mask_nodes

    def forward(self, x, h, edge_index, inference, **kwargs):
        # Recover mask_nodes passed from Model.forward
        mask_nodes = kwargs.get("mask_nodes", None)

        # The reconstruction is a training signal only: MAGIC scores nodes with its k-NN
        # detector, so the decoder is not run at inference
        if inference:
            return {"loss": torch.zeros((x.shape[0],), device=h.device)}

        if mask_nodes is None:
            raise ValueError(
                "GMAEFeatReconstruction objective requires masked nodes, but mask_nodes is None."
            )

        batch = kwargs.get("batch")
        edge_feats = getattr(batch, "edge_feats", None) if batch else None
        recon = self.decoder(h, edge_index, edge_feats=edge_feats)

        # Loss only on the masked nodes
        loss = self.loss_fn(recon[mask_nodes], x[mask_nodes].to(h.device), inference=False)
        return {"loss": loss}
