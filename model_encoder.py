import random
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from spatial_layer import DiffusionSpatialLayer, DistanceSpatialLayer, GCNLayer
from temporal_layer import RelativeTemporalLayer

import pdb

class EncoderModel(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.device = device
        self.diffusion_spatial_layer = DiffusionSpatialLayer(input_size, output_size, device)
        self.distance_spatial_layer = DistanceSpatialLayer(input_size, output_size, device)
        self.gcn_layer = GCNLayer(output_size, output_size, device)
        self.temporal_layer = RelativeTemporalLayer(input_size, output_size, device)
        self.random_layer = nn.Linear(input_size, output_size).to(self.device)

    def forward(self, feature_dg):
        nodes_embedding = []
        if len(feature_dg) == 1:
            random_values = feature_dg[0] # t * batch * nodes * 1
            random_tensor = torch.tensor(random_values).float().to(self.device)
            random_tensor = torch.unsqueeze(random_tensor, -1)
            nodes_time_embedding = torch.permute(self.random_layer(random_tensor), (1, 0, 2, 3)) # batch * t * nodes * dim
        else:
            spatial_embedding1 = self.diffusion_spatial_layer(feature_dg[0])
            spatial_embedding2 = self.distance_spatial_layer(feature_dg[1])
            temporal_embedding = self.temporal_layer(feature_dg[2])
            # batch * t * node * dim
            nodes_time_embedding = torch.sum(torch.stack([spatial_embedding1, spatial_embedding2, temporal_embedding]), 0)
        nodes_embedding = nodes_time_embedding.contiguous().view(
            nodes_time_embedding.shape[0], 
            nodes_time_embedding.shape[1] * nodes_time_embedding.shape[2], 
            nodes_time_embedding.shape[3])

        return nodes_embedding