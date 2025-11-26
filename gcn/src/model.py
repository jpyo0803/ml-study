import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, dropout=0.5):
        super().__init__()

        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_classes)
        self.dropout = dropout

    def forward(self, X, edge_index):
        X = self.conv1(X, edge_index)
        X = F.relu(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = self.conv2(X, edge_index)
        return F.log_softmax(X, dim=1)