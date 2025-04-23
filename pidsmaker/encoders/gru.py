import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, x_dim, h_dim, device, hidden_units=1):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(x_dim, h_dim, num_layers=hidden_units)
        self.hidden_units = hidden_units

        self.h_dim = h_dim
        self.device = device
        self.reset_state()

    def reset_state(self):
        self.hidden = self._init_hidden()

    def detach_state(self):
        self.hidden = self.hidden.detach()

    def _init_hidden(self):
        return torch.zeros(self.hidden_units, self.h_dim, requires_grad=True).to(self.device)

    def forward(self, xs, h0=None, include_h=False):
        xs, self.hidden = self.rnn(xs, self.hidden)

        if not include_h:
            return xs
        return xs, self.hidden
