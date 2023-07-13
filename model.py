import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np

from model_encoder import EncoderModel
from model_augment import ANAugmentModel
from model_transformer import TransformerModel
from model_rnn import RNNModel
from model_output import AnomalyDetectionModel, SequenceDecoderModel

import pdb

class Model(nn.Module):
    def __init__(self, args, an_layer_size, device):
        super().__init__()
        self.args = args
        self.device = device
        self.encoder = EncoderModel(1, args.embedding_output_dim, self.device)
        self.augmenter = ANAugmentModel(1, args.embedding_output_dim, an_layer_size, self.device)
        self.transformer = TransformerModel(
            args.embedding_output_dim, 
            args.embedding_output_dim, 
            args.transformer_output_dim, 
            args.attention_head_num,
            args.attention_layer_num,
            args.mask_encoder,
            args.num_context_node,
            args.time_window_len,
            args.trans_mode,
            args.pos_mode,
            self.device)
        self.rnn = RNNModel(
            args.embedding_output_dim, 
            args.embedding_output_dim, 
            args.time_window_len, 
            args.num_context_node, 
            self.device)
        self.detector = AnomalyDetectionModel(args.transformer_output_dim, 1, self.device)
        self.decoder = SequenceDecoderModel(args.transformer_output_dim, args.embedding_output_dim, self.device)

    def loss_function(self, score, label):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(torch.squeeze(score, 1), torch.tensor(label).float().to(self.device))
        return loss

    def loss_kl_divergence(self, input, target):
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        input = F.log_softmax(input, dim=2)
        target = F.softmax(target, dim=2)
        output = kl_loss(input, target)
        return output

    def evaluate(self, score, label):
        loss = self.loss_function(score, label)
        score = torch.squeeze(score, 1).cpu().numpy()
        label = np.array(label)
        auc = metrics.roc_auc_score(label, score)
        ap = metrics.average_precision_score(label, score)
        return auc, ap

    def forward(self, feature_dg, feature_an, feature_pos, label, mode):
        subgraph_embedding = self.encoder(feature_dg)

        augment_embedding = self.augmenter(feature_an)
        
        masked_subgraph_embedding = None
        edge_embedding, masked_subgraph_embedding = self.transformer(subgraph_embedding, augment_embedding, feature_pos, self.args.num_context_node, self.args.time_window_len)
        # edge_embedding = self.rnn(subgraph_embedding)

        if masked_subgraph_embedding != None:
            subgraph_output_embedding = self.decoder(masked_subgraph_embedding)
            loss_reconstruct = self.loss_kl_divergence(subgraph_embedding, subgraph_output_embedding)
        else:
            loss_reconstruct = 0.0

        anomaly_score = self.detector(edge_embedding)

        if mode == 'train':
            loss = self.loss_function(anomaly_score, label)
            return loss + loss_reconstruct
        elif mode == 'test':
            idx_sort = torch.argsort(torch.squeeze(anomaly_score, 1), descending=True)
            idx_topk = [idx_sort.tolist().index(i) for i in range(10)]
            auc, ap = self.evaluate(anomaly_score, label)
            return auc, ap, idx_topk
        else:
            from sklearn.decomposition import PCA
            edge_embedding = edge_embedding.cpu().numpy()
            pca = PCA(n_components = 2)
            pca.fit(edge_embedding)
            reduced_embedding = pca.transform(edge_embedding)
            with open('./embs/output_'+self.args.an_augment_mode+'_'+self.args.dataset, 'w') as f:
                f.write(str(reduced_embedding.tolist()))
            with open('./embs/label_'+self.args.dataset, 'w') as f:
                f.write(str(label))
            return None

