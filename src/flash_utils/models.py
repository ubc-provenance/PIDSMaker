import numpy as np
import torch
import math
from torch_geometric.nn import SAGEConv, GATConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv1 = SAGEConv(in_channel, 32, normalize=True)
        self.conv2 = SAGEConv(32, out_channel, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        return x

class PositionalEncoder:

    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]


def infer(document, w2vmodel, encoder):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]

    if not word_embeddings:
        return np.zeros(20)

    output_embedding = torch.tensor(word_embeddings, dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)