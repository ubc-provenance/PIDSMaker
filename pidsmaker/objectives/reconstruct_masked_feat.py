import torch
import torch.nn as nn


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
