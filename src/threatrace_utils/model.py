import torch
from torch_geometric.nn import SAGEConv, GATConv
import torch.nn.functional as F


class SAGENet(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super(SAGENet, self).__init__()
		self.conv1 = SAGEConv(in_channels, 32, normalize=False)
		self.conv2 = SAGEConv(32, out_channels, normalize=False)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = F.dropout(x, p=0.5, training=self.training)

		x = self.conv2(x, edge_index)

		return F.log_softmax(x, dim=1)