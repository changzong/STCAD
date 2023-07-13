import torch
import torch.nn as nn

class AnomalyDetectionModel(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.device = device
        self.scorer = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, input_emb):
        input_emb.to(self.device)
        score = self.scorer(input_emb)
        return score


class SequenceDecoderModel(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.device = device
        self.decoder = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Linear(output_size, output_size),
        ).to(self.device)

    def forward(self, input_emb):
        input_emb.to(self.device)
        output = self.decoder(input_emb)
        return output