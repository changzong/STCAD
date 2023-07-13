import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import networkx as nx

import pdb


class ANAugmentModel(nn.Module):
    def __init__(self, input_size, output_size, layer_size, device):
        super().__init__()
        self.device = device
        self.an_linears = []
        for i in range(layer_size):
            self.an_linears.append(nn.Linear(input_size, output_size).to(self.device))
        self.anomaly_mlp = nn.Sequential(
            nn.Linear(layer_size, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, feature_an):
        if len(feature_an) > 0:
            an_embeddings = []
            for i in range(len(feature_an)):
                an_tensor = torch.tensor(feature_an[i]).float().to(self.device)
                an_tensor = torch.unsqueeze(an_tensor, -1)
                an_embedding = self.an_linears[i](an_tensor)
                an_embeddings.append(an_embedding)
            an_embeddings = torch.stack(an_embeddings, 2)
            return torch.squeeze(self.anomaly_mlp(an_embeddings), 2) # batch * dim
        else:
            return None
 