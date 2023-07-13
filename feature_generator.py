import networkx as nx
import scipy as sp
import scipy.sparse
import numpy as np
import random

import pdb

def generate_sample_score_feature(input_data, snapshots):
    sample_values = []
    for t in range(len(snapshots)):
        input_sample_values = []
        for input_target in input_data:
            node_ids = input_target[0]+input_target[1][t]
            target_node1 = input_target[0][0]
            target_node2 = input_target[0][1]
            NE = snapshots[t].number_of_edges()
            tmp = []
            for node_id in node_ids:
                score = 1 / NE
                tmp.append(score)
            input_sample_values.append(tmp)
        sample_values.append(input_sample_values) # t * batch * node * 1
    return sample_values

def generate_prefer_attach_feature(input_data, snapshots):
    attach_values = []
    for t in range(len(snapshots)):
        input_attach_values = []
        for input_target in input_data:
            node_ids = input_target[0]+input_target[1][t]
            target_node1 = input_target[0][0]
            target_node2 = input_target[0][1]
            NE = snapshots[t].number_of_edges()
            N1 = len([n for n in snapshots[t].neighbors(target_node1)])
            N2 = len([n for n in snapshots[t].neighbors(target_node2)])
            tmp = []
            for node_id in node_ids:
                score =  (N1 * N2) / (NE * NE)
                tmp.append(score)
            input_attach_values.append(tmp)
        attach_values.append(input_attach_values) # t * batch * node * 1
    return attach_values

def generate_homophily_score_feature(input_data, snapshots):
    homophily_values = []
    for t in range(len(snapshots)):
        input_homophily_values = []
        for input_target in input_data:
            node_ids = input_target[0]+input_target[1][t]
            target_node1 = input_target[0][0]
            target_node2 = input_target[0][1]
            NE = snapshots[t].number_of_edges()
            N1 = [n for n in snapshots[t].neighbors(target_node1)]
            N2 = [n for n in snapshots[t].neighbors(target_node2)]
            common = len(list(set(N1) & set(N2)))
            tmp = []
            for node_id in node_ids:
                score =  (common * 2) / (NE * 2)
                tmp.append(score)
            input_homophily_values.append(tmp)
        homophily_values.append(input_homophily_values) # t * batch * node * 1
    return homophily_values

def generate_diffusion_spatial_feature(input_data, snapshots):
    diffusion_values = []
    for t in range(len(snapshots)):
        try:
            diffusion_value = nx.pagerank(snapshots[t], alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            diffusion_value = nx.pagerank(snapshots[t], alpha=0.85, tol=1.0e-3)
        tmp = []
        for input_target in input_data:
            node_ids = input_target[0]+input_target[1][t]
            diffusion_subgraph = {node_id: diffusion_value[node_id] for node_id in node_ids} 
            diffusion_subgraph_order = dict(sorted(diffusion_subgraph.items(), key=lambda item: item[1], reverse=True))
            node_id_list = list(diffusion_subgraph_order.keys())
            node_rank = [node_id_list.index(node_id)+1 for node_id in node_ids]
            tmp.append(node_rank)
        diffusion_values.append(tmp)  # t * batch * node * 1
    return diffusion_values

def generate_distance_spatial_feature(input_data, snapshots):
    distance_values = []
    for t in range(len(snapshots)):
        input_distance_values = []
        for input_target in input_data:
            node_ids = input_target[0]+input_target[1][t]
            target_node1 = input_target[0][0]
            target_node2 = input_target[0][1]
            tmp = []
            for node_id in node_ids:
                try:
                    distance1 = nx.shortest_path_length(snapshots[t], source=node_id, target=target_node1)
                except nx.NetworkXNoPath:
                    distance1 = 100
                try:
                    distance2 = nx.shortest_path_length(snapshots[t], source=node_id, target=target_node2)
                except nx.NetworkXNoPath:
                    distance2 = 100
                tmp.append(min(distance1, distance2))
            input_distance_values.append(tmp)
        distance_values.append(input_distance_values) # t * batch * node * 1
    return distance_values

def generate_relative_temporal_feature(input_data, snapshots):
    temporal_values = []
    for t in range(len(snapshots)):
        occur_time = 0
        current_time = t
        tmp = []
        for input_target in input_data:
            node_ids = input_target[0]+input_target[1][t]
            target_node1 = input_target[0][0]
            target_node2 = input_target[0][1]
            for i in range(current_time):
                subgraph = input_target[2][i]
                if subgraph.has_edge(target_node1, target_node2):
                    occur_time = i
                    break
            node_temporal = [abs(current_time - occur_time)] * len(node_ids)
            tmp.append(node_temporal)
        temporal_values.append(tmp) # t * batch * node * 1
    return temporal_values

def generate_distance_steepness_feature(input_data, snapshots):
    steepness_scores = []
    for input_target in input_data:
        target_node1 = input_target[0][0]
        target_node2 = input_target[0][1]
        distance_now = 1
        try:
            distance_prev = nx.shortest_path_length(snapshots[-2], source=target_node1, target=target_node2)
        except nx.NetworkXNoPath:
            distance_prev = 100
        steepness_score = abs(distance_prev - distance_now)
        steepness_scores.append(steepness_score) # batch * 1
    return steepness_scores

def generate_neighbor_interact_feature(input_data, snapshots):
    frequency_scores = []
    for input_target in input_data:
        target_node1 = input_target[0][0]
        target_node2 = input_target[0][1]
        # interact1 = snapshots[-2].degree(target_node1) + snapshots[-1].degree(target_node1)
        # interact2 = snapshots[-2].degree(target_node2) + snapshots[-1].degree(target_node2)
        interact1 = snapshots[-2].degree(target_node1) + snapshots[-2].degree(target_node2)
        interact2 = snapshots[-1].degree(target_node1) + snapshots[-1].degree(target_node2)
        # interact_score = interact1 + interact2
        interact_score = abs(interact1 - interact2)
        frequency_scores.append(interact_score) # batch * 1
    return frequency_scores

def generate_common_neighbor_feature(input_data, snapshots):
    common_scores = []
    for input_target in input_data:
        target_node1 = input_target[0][0]
        target_node2 = input_target[0][1]
        neighbor_node11 = [n for n in snapshots[-2].neighbors(target_node1)]
        neighbor_node12 = [n for n in snapshots[-2].neighbors(target_node2)]
        comm_score1 = len(list(set(neighbor_node11) & set(neighbor_node12)))
        neighbor_node21 = [n for n in snapshots[-1].neighbors(target_node1)]
        neighbor_node22 = [n for n in snapshots[-1].neighbors(target_node2)]
        comm_score2 = len(list(set(neighbor_node21) & set(neighbor_node22)))
        # common_score = len(list(set(neighbor_node1) & set(neighbor_node2)))
        common_score = abs(comm_score2 - comm_score1)
        common_scores.append(common_score) # batch * 1
    return common_scores

def generate_random_feature(input_data, snapshots):
    random_values = []
    for t in range(len(snapshots)):
        input_random_values = []
        for input_target in input_data:
            node_ids = input_target[0]+input_target[1][t]
            tmp = []
            for node_id in node_ids:
                tmp.append(random.random())
            input_random_values.append(tmp)
        random_values.append(input_random_values)
    return random_values # t * batch * nodes * 1

def generate_absolute_positional_encoding(input_data, snapshots):
    absolute_values = []
    for t in range(len(snapshots)):
        absolute_value = nx.adjacency_spectrum(snapshots[t])
        tmp = []
        for input_target in input_data:
            node_ids = input_target[0]+input_target[1][t]
            node_pe = [absolute_value[node_id - 1] for node_id in node_ids]
            tmp.append(node_pe)
        absolute_values.append(tmp)  # t * batch * node * 1
    return absolute_values

""" def generate_absolute_positional_encoding(input_data, snapshots):
    absolute_values = []
    for t in range(len(snapshots)):
        A = nx.to_scipy_sparse_array(snapshots[t], format="csr")
        n, m = A.shape
        D = sp.sparse.csr_array(sp.sparse.spdiags(A.sum(axis=1), 0, m, n, format="csr"))
        RW = A.todense() * (D.todense().clip(1) ** -1.0)
        M = RW
        RW_pe = []
        for k in range(10):
            M = M * RW
            RW_pe.append(M.diagonal())
        # absolute_value = nx.adjacency_spectrum(snapshots[t])
        tmp = []
        for input_target in input_data:
            node_ids = input_target[0]+input_target[1][t]
            node_pe = [[pe[node_id-1] for pe in RW_pe] for node_id in node_ids]
            tmp.append(node_pe)
        absolute_values.append(tmp)  # t * batch * node * 1
    return absolute_values """

def generate_relative_positional_encoding(input_data, snapshots):
    absolute_values = []
    for t in range(len(snapshots)):
        tmp = []
        for input_target in input_data:
            node_ids = input_target[0]+input_target[1][t]
            neighbors1 = [n for n in snapshots[t].neighbors(node_ids[0])]
            neighbors2 = [n for n in snapshots[t].neighbors(node_ids[1])]
            common_neighbors = list(set(neighbors1) & set(neighbors2))
            pos_relative = [0,0]
            for node in node_ids[2:]:
                if node in common_neighbors:
                    pos_relative.append(1)
                else:
                    pos_relative.append(2)
            tmp.append(pos_relative)
        absolute_values.append(tmp)  # t * batch * node * 1
    return absolute_values

def generate_features(input_data, snapshots, dg_augment_mode, an_augment_mode, pos_mode):
    if dg_augment_mode == 's':
        dg_feature1 = generate_sample_score_feature(input_data, snapshots)
        dg_feature2 = generate_prefer_attach_feature(input_data, snapshots)
        dg_feature3 = generate_homophily_score_feature(input_data, snapshots)
        dg_feature = [dg_feature1, dg_feature2, dg_feature3]
    elif dg_augment_mode == 'st':
        dg_feature1 = generate_diffusion_spatial_feature(input_data, snapshots)
        dg_feature2 = generate_distance_spatial_feature(input_data, snapshots)
        dg_feature3 = generate_relative_temporal_feature(input_data, snapshots)
        dg_feature = [dg_feature1, dg_feature2, dg_feature3]
    elif dg_augment_mode == 'no':
        dg_feature = [generate_random_feature(input_data, snapshots)]
    else:
        return NotImplemented
    
    if an_augment_mode == 'all':
        an_feature1 = generate_distance_steepness_feature(input_data, snapshots)
        an_feature2 = generate_neighbor_interact_feature(input_data, snapshots)
        an_feature3 = generate_common_neighbor_feature(input_data, snapshots)
        an_feature = [an_feature1, an_feature2, an_feature3]
    elif an_augment_mode == 'distance':
        an_feature = [generate_distance_steepness_feature(input_data, snapshots)]
    elif an_augment_mode == 'interact':
        an_feature = [generate_neighbor_interact_feature(input_data, snapshots)]
    elif an_augment_mode == 'common':
        an_feature = [generate_common_neighbor_feature(input_data, snapshots)]
    elif an_augment_mode == 'no':
        an_feature = []
    else:
        return NotImplemented

    if pos_mode == '2d' or pos_mode == 'spos':
        pos_feature1 = generate_absolute_positional_encoding(input_data, snapshots)
        pos_feature2 = generate_relative_positional_encoding(input_data, snapshots)
        pos_feature = [pos_feature1, pos_feature2]
    else:
        pos_feature = None

    return dg_feature, an_feature, pos_feature