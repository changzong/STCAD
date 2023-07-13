import torch
import torch.nn as nn
import random

import pdb

from transformer_layer import TransformerEncoderLayer, PoolingLayer

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_head, num_layer, mask_encoder, context_num, time_length, trans_mode, pos_mode, device):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.mask_encoder = mask_encoder
        self.transformer = TransformerEncoderLayer(time_length * context_num, input_size, hidden_size, output_size, num_head, num_layer, context_num, trans_mode, device)
        self.pooling = PoolingLayer(input_size, output_size)
        self.hidden_size = hidden_size
        self.pos_mode = pos_mode
        if pos_mode == '2d':
            self.pos_linear_w1 = nn.Linear(1, int(input_size / 2)).to(self.device)
            self.pos_linear_w2 = nn.Linear(1, int(input_size / 2)).to(self.device)
            self.pos_linear_h = nn.Linear(1, input_size).to(self.device)
            self.pos_emb_t = self.get_temporal_positional_embedding(context_num, time_length)
        elif pos_mode == 'tpos':
            self.pos_linear = nn.Linear(1, input_size).to(self.device)
            self.pos_emb = self.get_temporal_positional_embedding(context_num, time_length)
        elif pos_mode == 'spos':
            self.pos_linear_w1 = nn.Linear(10, int(input_size / 2)).to(self.device)
            self.pos_linear_w2 = nn.Linear(1, int(input_size / 2)).to(self.device)

    def get_masked_embedding(self, subgraph_embedding, mask_pos):
        mask = torch.zeros(self.input_size)
        subgraph_embedding[:,mask_pos] = mask
        return subgraph_embedding

    def get_simple_positional_embedding(self, context_num, time_length):
        pos_emb = []
        for i in range(time_length):
            for j in range(context_num+2):
                tmp = [i, j]
                pos_emb.append(tmp)
        return torch.tensor(pos_emb).float().to(self.device)

    def get_temporal_positional_embedding(self, context_num, time_length):
        pos_emb = []
        for i in range(time_length):
            for j in range(context_num+2):
                pos_emb.append(i)
        return torch.tensor(pos_emb).float().to(self.device)

    def get_spatial_positional_embedding(self, feature_pos, pos_linear):
        pos_tensor = torch.tensor(feature_pos).float().to(self.device)
        pos_embeddings = []
        for i in range(len(feature_pos)):
            pos_tensor = torch.tensor(feature_pos[i]).float().to(self.device)
            if len(pos_tensor.size()) == 2:
                pos_tensor = torch.unsqueeze(pos_tensor, -1)
            pos_embedding = pos_linear(pos_tensor)
            pos_embeddings.append(pos_embedding)
        pos_embeddings = torch.stack(pos_embeddings, 2)
        return pos_embeddings.contiguous().view(pos_embeddings.shape[0], pos_embeddings.shape[1]*pos_embeddings.shape[2], pos_embeddings.shape[3])

    def forward(self, subgraph_embedding, augment_embedding, feature_pos, context_num, time_length):
        subgraph_embedding.to(self.device)
        if self.mask_encoder == 'simple':
            mask_pos = random.randint(0, subgraph_embedding.size(1)-1) # randomly select mask position
            masked_subgraph_embedding, _ = self.transformer(self.get_masked_embedding(subgraph_embedding, mask_pos))
        elif self.mask_encoder == 'no':
            masked_subgraph_embedding = None

        if self.pos_mode == '2d':
            pos_emb_w1 = self.get_spatial_positional_embedding(feature_pos[0], self.pos_linear_w1)
            pos_emb_w2 = self.get_spatial_positional_embedding(feature_pos[1], self.pos_linear_w2)
            pos_emb_h = self.pos_linear_h(torch.unsqueeze(self.pos_emb_t, -1)).repeat(subgraph_embedding.size()[0],1,1)
            pos_emb_w = torch.cat((pos_emb_w1, pos_emb_w2), 2)
            positional_embedding = pos_emb_w + pos_emb_h
            subgraph_embedding = subgraph_embedding + positional_embedding
        elif self.pos_mode == 'tpos':
            positional_embedding = self.pos_linear(torch.unsqueeze(self.pos_emb, -1))
            subgraph_embedding = subgraph_embedding + positional_embedding.repeat(subgraph_embedding.size()[0],1,1)
        elif self.pos_mode == 'spos':
            pos_emb_w1 = self.get_spatial_positional_embedding(feature_pos[0], self.pos_linear_w1)
            pos_emb_w2 = self.get_spatial_positional_embedding(feature_pos[1], self.pos_linear_w2)
            positional_embedding = torch.cat((pos_emb_w1, pos_emb_w2), 2)
            subgraph_embedding = subgraph_embedding + positional_embedding
        
        if augment_embedding != None:
            tmp1 = [augment_embedding, augment_embedding] # two target nodes
            tmp2 = [torch.zeros_like(augment_embedding) for i in range(context_num)]
            tmp = torch.permute(torch.stack(tmp1 + tmp2), (1,0,2))
            subgraph_embedding, _ = self.transformer(subgraph_embedding + tmp.repeat(1,time_length,1)) # batch * t * nodes * dim
        else:
            subgraph_embedding, _ = self.transformer(subgraph_embedding)
            # subgraph_embedding = subgraph_embedding # for feature only test
        
        edge_embedding = self.pooling(subgraph_embedding)
        
        return edge_embedding, masked_subgraph_embedding