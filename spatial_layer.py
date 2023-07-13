import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import networkx as nx

import pdb

class DiffusionSpatialLayer(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(input_size, output_size).to(self.device)

    def forward(self, diffusion_values):
        diffusion_tensor = torch.tensor(diffusion_values).float().to(self.device)
        diffusion_tensor = torch.unsqueeze(diffusion_tensor, -1)
        diffusion_embedding = torch.permute(self.linear(diffusion_tensor), (1, 0, 2, 3)) # batch * t * nodes * dim
        return diffusion_embedding


class DistanceSpatialLayer(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(input_size, output_size).to(self.device)

    def forward(self, distance_values):
        distance_tensor = torch.tensor(distance_values).float().to(self.device)
        distance_tensor = torch.unsqueeze(distance_tensor, -1)
        distance_embedding = torch.permute(self.linear(distance_tensor), (1, 0, 2, 3)) # batch * t * nodes * dim
        return distance_embedding


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device, dropout=0.2):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = F.relu
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim)).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def sparse_mx_to_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        values = sparse_mx.data
        indices = np.vstack((sparse_mx.row, sparse_mx.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = sparse_mx.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, inputs, snapshots):
        embeddings = []
        for t in range(inputs.shape[0]):
            input_d = F.dropout(inputs[t], self.dropout, self.training).to(self.device)
            support = torch.mm(input_d, self.weight)
            sparse_adj_matrix = nx.adjacency_matrix(snapshots[t])
            sparse_adj_tensor = self.sparse_mx_to_tensor(sparse_adj_matrix).to(self.device)
            output = torch.spmm(sparse_adj_tensor, support)
            output = self.act(output)
            embeddings.append(output)
        return torch.stack(embeddings) # t * all_nodes * dim