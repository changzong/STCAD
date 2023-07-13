import torch
import torch.nn as nn
import numpy as np

import pdb

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.hidden_size)     
        attention = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attention, V)
        return context, attention

# How Much Does Attention Actually Attend
# https://arxiv.org/abs/2211.03495v1
class ConstantDotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, Q, K, V):
        attention = None # Constant to be computed
        context = torch.matmul(attention, V)
        return context, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_head, device):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.W_Q = nn.Linear(input_size, hidden_size * num_head, bias=False).to(self.device)
        self.W_K = nn.Linear(input_size, hidden_size * num_head, bias=False).to(self.device)
        self.W_V = nn.Linear(input_size, hidden_size * num_head, bias=False).to(self.device)
        self.fc = nn.Linear(num_head * hidden_size, input_size, bias=False).to(self.device)
        self.attention_layer = ScaledDotProductAttention(input_size)
        self.norm_layer = nn.LayerNorm(input_size).to(self.device)

    def forward(self, input_Q, input_K, input_V):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.num_head, self.hidden_size).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, self.num_head, self.hidden_size).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, self.num_head, self.hidden_size).transpose(1,2)

        context, attention = self.attention_layer(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_head * self.hidden_size)
        output = self.fc(context)
        return self.norm_layer(output + residual), attention

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size, bias=False),
            nn.ReLU(),
            nn.Linear(output_size, input_size, bias=False)
        ).to(self.device)
        self.norm_layer = nn.LayerNorm(input_size).to(self.device)
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.norm_layer(output + residual) 

class EncoderLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_head, num_context, trans_mode, device):
        super().__init__()
        self.device = device
        self.multihead_attention = MultiHeadAttention(input_size, hidden_size, num_head, self.device)
        self.ffn = PoswiseFeedForwardNet(input_size, output_size, self.device)
        self.num_context = num_context
        self.trans_mode = trans_mode

    def forward(self, inputs):
        if self.trans_mode == 'target':
            index_list = []
            for item in range(0, inputs.size()[1], self.num_context+2):
                index_list.append(item)
                index_list.append(item+1)
            query = torch.index_select(inputs, 1, torch.tensor(index_list))
        else:
            query = inputs
        outputs, attention = self.multihead_attention(query, inputs, inputs)
        outputs = self.ffn(outputs)
        return outputs, attention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_size, num_head, num_layer, num_context, trans_mode, device):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            if i == 0:
                self.layers.append(EncoderLayer(input_size, hidden_size, output_size, num_head, num_context, trans_mode, self.device))
            else:
                self.layers.append(EncoderLayer(input_size, hidden_size, output_size, num_head, num_context, 'self', self.device))

    def forward(self, inputs):
        inputs.to(self.device)
        attentions = []
        outputs = inputs
        for layer in self.layers:
            outputs, attention = layer(outputs)
            attentions.append(attention)
        return outputs, attentions

class PoolingLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

    def forward(self, input_emb):
        # input_emb: batch * (t*nodes) * dim
        return torch.mean(input_emb, 1) # batch * dim
