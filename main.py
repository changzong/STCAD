import torch
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

from sampling import *
from feature_generator import *
from model import Model

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['uci', 'digg', 'email-dnc', 'btc-alpha', 'btc-otc', 'as-topology', 'ai-patent', 'nci-project'], default='uci')
parser.add_argument('--embedding_output_dim', type=int, default=32)
parser.add_argument('--transformer_output_dim', type=int, default=32)
parser.add_argument('--num_context_node', type=int, default=5)
parser.add_argument('--time_window_len', type=int, default=4)
parser.add_argument('--context_strategy', type=str, choices=['hop-1', 'shared-neighbor', 'diffusion'], default='hop-1')
parser.add_argument('--attention_layer_num', type=int, default=2)
parser.add_argument('--attention_head_num', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--valid_ratio', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--anomaly_ratio', type=float, default=0.1)
parser.add_argument('--epoch_num', type=int, default=300)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--training_ratio', type=int, default=1.0)
parser.add_argument('--trans_mode', type=str, choices=['self', 'target'], default='self')
parser.add_argument('--an_augment_mode', type=str, choices=['no', 'distance', 'interact', 'common', 'all'], default='all')
parser.add_argument('--dg_augment_mode', type=str, choices=['no', 's', 'st'], default='st')
parser.add_argument('--mask_encoder', type=str, choices=['no', 'simple'], default='simple')
parser.add_argument('--pos_mode', type=str, choices=['no', '2d', 'tpos', 'spos'], default='2d')
parser.add_argument('--use_cuda', action='store_true', default=True)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

train_dataset, test_dataset = edge_based_sampling(args.dataset, args.anomaly_ratio, args.num_context_node, args.context_strategy, args.time_window_len, args.training_ratio)
dataset, labels, snapshots = train_dataset
dataset_train, dataset_valid, labels_train, label_valid = train_test_split(dataset, labels, test_size=args.valid_ratio, random_state=args.seed)
dataset_test, labels_test, snapshots_test = test_dataset

print('Generating knowledge augmented features...')
train_feature_dg, train_feature_an, train_feature_pos = generate_features(dataset_train, snapshots, args.dg_augment_mode, args.an_augment_mode, args.pos_mode)
test_feature_dg, test_feature_an, test_feature_pos = generate_features(dataset_test, snapshots_test, args.dg_augment_mode, args.an_augment_mode, args.pos_mode)

model = Model(args, len(train_feature_an), device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
print('train normal size: %s train anomaly size: %s' % (str(len([l for l in labels_train if l == 0])), str(len([l for l in labels_train if l == 1]))))
best_auc = 0.0
best_ap = 0.0
for epoch in range(args.epoch_num):
    model.train()
    loss = model(train_feature_dg, train_feature_an, train_feature_pos, labels_train, 'train')
    print('Epoch: %s Training loss: %s' % (str(epoch), str(loss.item() / len(dataset_train))))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('test normal size: %s test anomaly size: %s' % (str(len([l for l in labels_test if l == 0])), str(len([l for l in labels_test if l == 1]))))
        with torch.no_grad():
            model.eval()
            auc, ap, topk = model(test_feature_dg, test_feature_an, test_feature_pos, labels_test, 'test')
            target_topk = [dataset_test[idx][0] for idx in topk]
            if auc > best_auc:
                best_auc = auc
            if ap > best_ap:
                best_ap = ap
            print('test AUC: %s' % str(auc))
            print('test AP: %s' % str(ap))
            print('Edge with the highest score: %s' % str(target_topk))
    if (epoch + 1) % 500 == 0:
        with torch.no_grad():
            model.eval()
            model(test_feature_dg, test_feature_an, labels_test, 'reduce')
            print('Output dimension reduction embedding')

print('Best results: AUC: %s, AP: %s' % (str(best_auc), str(best_ap)))
            
        
            