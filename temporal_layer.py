import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pdb

class RelativeTemporalLayer(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(input_size, output_size).to(self.device)

    def forward(self, relative_values):
        relative_tensor = torch.tensor(relative_values).float().to(self.device)
        relative_tensor = torch.unsqueeze(relative_tensor, -1)
        relative_embedding = torch.permute(self.linear(relative_tensor), (1, 0, 2, 3)) # batch * t * nodes * dim
        return relative_embedding