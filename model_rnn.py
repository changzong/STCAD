import torch
import torch.nn as nn

import pdb

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, time_length, context_num, device):
        super().__init__()
        self.device = device
        self.time_length = time_length
        self.context_num = context_num
        self.rnn = nn.GRU(input_size, output_size, batch_first=True, bidirectional=True).to(self.device)
        self.fc = nn.Linear(2 * output_size, output_size).to(self.device)
        self.relu = nn.ReLU()
        for name, param in self.rnn.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, input_embeds):
        input_embeds = input_embeds.view(
            input_embeds.shape[0], 
            self.time_length,
            self.context_num + 2, 
            input_embeds.shape[2])
        input_embeds = torch.mean(input_embeds, 2) # batch * time * dim
        output, final_state = self.rnn(input_embeds) # 
        final_state = self.fc(self.relu(torch.cat([final_state[-1], final_state[-2]], dim=1)))
        return final_state # batch * dim
