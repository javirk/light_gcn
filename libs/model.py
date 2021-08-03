import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Model(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(Model, self).__init__()
        self.conv1 = GCNConv(input_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, output_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x