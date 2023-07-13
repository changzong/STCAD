import networkx as nx
import random
import numpy as np
import scipy as sp
import scipy.sparse 
from datetime import datetime

import pdb

random.seed(1)

snapshot_size_dict = {'uci': 4000, 'digg': 6000, 'email-dnc':3000, 'as-topology': 8000, 'btc-alpha': 2000, 'btc-otc': 2000, 'ai-patent': 8000, 'nci-project': 8000}

def read_data_file(dataset):
    print('Reading files...')
    nodes = [] # list of node ids
    edges = [] # list of edge tuples with (head, tail, weight, timestamp)
    with open('./dataset/raw/'+dataset, 'r') as f:
        lines = f.readlines()
    content = lines[2:]
    for line in content:
        tmp = line.strip().split(' ')
        if int(tmp[0]) not in nodes:
            nodes.append(int(tmp[0]))
        if int(tmp[1]) not in nodes:
            nodes.append(int(tmp[1]))
        edges.append((int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])))
    return nodes, edges

def split_graph_by_time(nodes, edges, time_window_len):
    print('Generating snapshots...')
    snapshots = []
    time_max = datetime.fromtimestamp(max([edge[3] for edge in edges]))
    time_min = datetime.fromtimestamp(min([edge[3] for edge in edges]))
    interval = (time_max - time_min) / time_window_len
    for i in range(time_window_len):
        G = nx.MultiGraph() # treat snapshots as undirected multigraphs
        G.add_nodes_from(nodes) # assume all nodes are already in each snapshot
        for edge in edges:
            current_time = datetime.fromtimestamp(edge[3])
             # use accumulative snapshot
            if current_time >= time_min and current_time < (time_min + (i+1) * interval):
                G.add_edge(edge[0], edge[1], weight=edge[2])
            else:
                continue
        snapshots.append(G)
    return snapshots

# need to sort edge by timestamp first later!!!
def split_graph_by_size(nodes, edges, dataset_name):
    print('Generating snapshots...')
    dtype = [('head', int),('tail', int),('weight', int), ('time',int)]
    edges = np.array(edges, dtype=dtype)
    edges = list(np.sort(edges, order='time'))
    snapshots = []
    count = 0
    G = nx.MultiGraph()
    G.add_nodes_from(nodes)
    for edge in edges:
        if count < snapshot_size_dict[dataset_name]:
            G.add_edge(edge[0], edge[1], weight=edge[2])
            count += 1
        else:
            count = 0
            snapshots.append(G)
            G = nx.MultiGraph()
            G.add_nodes_from(nodes)
    return snapshots

def negative_node_pair_sampling(snapshots, sample_size):
    nodes_current = [n for n in snapshots[-1].nodes]
    node_pairs = []
    while(1):
        candidate = random.sample(nodes_current, 2)
        for i in range(len(snapshots)):
            if snapshots[i].has_edge(candidate[0], candidate[1]):
                break
            else:
                continue
        if i == len(snapshots) - 1 and candidate not in node_pairs:
            node_pairs.append(candidate)
        if len(node_pairs) == sample_size:
            break
    return node_pairs

def sampling_strategy(snapshots, context_strategy, num_context_node, time_window_len, anomaly_ratio, training_ratio):
    edges_current = [[e[0], e[1]] for e in snapshots[-1].edges]  # sample target edges from current timestamp
    target_edges_normal = random.sample(edges_current, round(len(edges_current) * training_ratio))
    target_edges_anomaly = negative_node_pair_sampling(snapshots, round(len(target_edges_normal) * anomaly_ratio))
    target_edges = target_edges_normal + target_edges_anomaly
    labels = [0] * len(target_edges_normal) + [1] * len(target_edges_anomaly)
    if context_strategy == 'hop-1':
        context_nodes = hop_1_context_sampling(target_edges, num_context_node, snapshots, time_window_len)
    elif context_strategy == 'shared-neighbor':
        context_nodes = shared_neighbor_context_sampling(target_edges, num_context_node, snapshots, time_window_len)
    elif context_strategy == 'diffusion':
        context_nodes = diffusion_context_sampling(target_edges, num_context_node, snapshots, time_window_len)
    else:
        raise Exception('Sampling strategy not implemented.')
    context_nodes_neg = random_sampling(target_edges_normal, num_context_node, snapshots, time_window_len)
    model_input = []
    for i in range(len(target_edges)):
        subgraphs = []
        for t in range(time_window_len):
            sub_nodes = context_nodes[i][t] + target_edges[i]
            subgraph = snapshots[t].subgraph(sub_nodes)
            subgraphs.append(subgraph)
        model_input.append([target_edges[i], context_nodes[i], subgraphs])
    return (model_input, labels, snapshots)

def edge_based_sampling(dataset_name, anomaly_ratio, num_context_node, context_strategy, time_window_len, training_ratio):
    nodes, edges = read_data_file(dataset_name)
    ## snapshots = split_graph_by_time(nodes, edges, time_window_len * 2)
    # snapshot_len = round(len(edges) / snapshot_size_dict[dataset_name])
    # snapshots = split_graph_by_time(nodes, edges, snapshot_len)
    snapshots = split_graph_by_size(nodes, edges, dataset_name)
    snapshots_train = snapshots[:time_window_len]
    snapshots_test = snapshots[-time_window_len:]
    train_dataset = sampling_strategy(snapshots_train, context_strategy, num_context_node, time_window_len, 1.0, training_ratio)
    test_dataset = sampling_strategy(snapshots_test, context_strategy, num_context_node, time_window_len, anomaly_ratio, 1.0)
    return train_dataset, test_dataset

# strategy-1: 1-hop neighbors of target nodes as context nodes
def hop_1_context_sampling(target_edges, num_context_node, snapshots, time_window_len):
    print('Sampling context nodes...')
    context_nodes = []
    for edge in target_edges:
        context_snapshot = []
        for i in range(time_window_len):
            context_node1 = [n for n in snapshots[i].neighbors(edge[0])]
            context_node2 = [n for n in snapshots[i].neighbors(edge[1])]
            context_all = list(set(context_node1) | set(context_node2))
            if len(context_all) >= num_context_node:
                context_select = random.sample(context_all, num_context_node)
            else:
                # using target node as context node for completion
                context_select = context_all + [edge[0]]*(num_context_node - len(context_all))
            context_snapshot.append(context_select)
        context_nodes.append(context_snapshot)
    return context_nodes

# strategy-2: shared neighbors of target nodes as context nodes
def shared_neighbor_context_sampling(target_edges, num_context_node, snapshots, time_window_len):
    print('Sampling context nodes...')
    context_nodes = []
    for edge in target_edges:
        context_snapshot = []
        for i in range(time_window_len):
            context_node1 = [n for n in snapshots[i].neighbors(edge[0])]
            context_node2 = [n for n in snapshots[i].neighbors(edge[1])]
            context_all = list(set(context_node1) & set(context_node2))
            if len(context_all) >= num_context_node:
                context_select = random.sample(context_all, num_context_node)
            else:
                context_select = context_all + [edge[0]]*(num_context_node - len(context_all))
            context_snapshot.append(context_select)
        context_nodes.append(context_snapshot)
    return context_nodes

# strategy-3: connectivity from diffusion matrix of target edges as context nodes
def diffusion_context_sampling(target_edges, num_context_node, snapshots, time_window_len):
    print('Sampling context nodes...') # too slow
    context_nodes = []
    alpha = 0.85
    identity_matrix = np.identity(snapshots[0].number_of_nodes())
    count = 0
    for edge in target_edges:
        context_snapshot = []
        for i in range(time_window_len):
            diff_matrix = alpha * (identity_matrix - (1-alpha) * nx.normalized_laplacian_matrix(snapshots[i]))
            conn_node1 = diff_matrix[edge[0] - 1].A1 # minus 1 to get row index
            conn_node2 = diff_matrix[edge[1] - 1].A1
            conn_edge = conn_node1 + conn_node2
            context_row_idx = (-conn_edge).argsort()[1:num_context_node+1] # get top k except target node itself
            context_all = context_row_idx + 1 # plus 1 to get node id
            context_snapshot.append(list(context_all))
        context_nodes.append(context_snapshot)
        count += 1
        print(count)
    return context_nodes

# strategy-4: randome selection
def random_sampling(target_edges, num_context_node, snapshots, time_window_len):
    print('Sampling random nodes...')
    context_nodes = []
    for edge in target_edges:
        context_snapshot = []
        for i in range(time_window_len):
            context_select = random.sample([n for n in snapshots[i].nodes], num_context_node)
            context_snapshot.append(context_select)
        context_nodes.append(context_snapshot)
    return context_nodes

# Contrast-1: Target edge should be closer to its context subgraphs than its unrelated subgraphs
def edge_based_context_sampling(dataset_name, sample_size, num_context_node, context_strategy='hop-1', time_window_len=5):
    nodes, edges = read_data_file(dataset_name)
    snapshots = split_graph_by_time(nodes, edges, time_window_len)
    edges_current = [[e[0], e[1]] for e in snapshots[-1].edges]
    target_edges = random.sample(edges_current, sample_size)
    # context sampling, using one-hop for now, can be replace with graph diffusion
    if context_strategy == 'hop-1':
        nodes_context = hop_1_context_sampling(target_edges, num_context_node, snapshots, time_window_len)
    else:
        raise Exception('Sampling strategy not implemented.')
    # unrelated sampling, using random selection
    nodes_unrelated = random_sampling(target_edges, num_context_node, snapshots, time_window_len)
    nodes = nodes_context + nodes_unrelated
    edges = target_edges + target_edges
    labels = [1] * sample_size + [0] * sample_size
    model_input = []
    for i in range(len(edges)):
        subgraphs = []
        for t in range(time_window_len):
            sub_nodes = nodes[i][t] + edges[i]
            subgraph = snapshots[t].subgraph(sub_nodes)
            subgraphs.append(subgraph)
        model_input.append([edges[i], nodes[i], subgraphs])
    return model_input, labels, snapshots

# Contrast-2: Target edge should be closer to its same view subgraph than its cross-view subgraph
def edge_based_view_sampling():
    pass

# Contrast-3: Target edge at a timestamp should be closer to its context timestamp than its unrelated timestamp
def edge_based_timestamp_sampling(seq_len, ref_len_limit=3, target_len_limit=2):
    timestamp_samples = []
    ref_range = list(range(0, seq_len - ref_len_limit + 1))
    for start in ref_range:
        ref_seq = list(range(start, start + ref_len_limit))
        pos_range = list(range(0, ref_len_limit - target_len_limit + 1))
        for pos_start in pos_range:
            pos_seq = list(range(ref_seq[pos_start], ref_seq[pos_start] + target_len_limit))
            if ref_seq[0] > target_len_limit:
                neg_range = list(range(0, ref_seq[0] + 1 - target_len_limit + 1))
                for neg_start in neg_range:
                    neg_seq = list(range(neg_start, neg_start + target_len_limit))
                    timestamp_samples.append([ref_seq, pos_seq, neg_seq])
            elif ref_seq[-1] < seq_len - target_len_limit:
                neg_range = list(range(ref_seq[-1] + 1, seq_len))
                for neg_start in neg_range:
                    if neg_start + target_len_limit <= seq_len:
                        neg_seq = list(range(neg_start, neg_start + target_len_limit))
                        timestamp_samples.append([ref_seq, pos_seq, neg_seq])
            else:
                break
    return timestamp_samples


if __name__ == '__main__':
    # model_input, labels, snapshots = edge_based_sampling('uci', 1000, 5, context_strategy='hop-1', time_window_len=5)
    model_input, labels, snapshots = edge_based_context_sampling('uci', 1000, 5, context_strategy='hop-1', time_window_len=5)
    timestamp_samples = edge_based_timestamp_sampling(5, ref_len_limit=3, target_len_limit=2)
    pdb.set_trace()