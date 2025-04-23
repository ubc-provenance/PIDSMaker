import torch
import torch.nn as nn
import torch.nn.functional as F


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
