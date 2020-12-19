import numpy as np
import igraph as ig 
import csv
import numpy as np
import pandas as pd
import os
from graspologic.simulations import sbm

def load_adj(path):
    full_path = os.path.join(path, 'adj.txt')
    num_nodes = -1
    adj = []
    with open(full_path, mode='r') as txt_file:
        for row in txt_file:
            row = row.split(",")
            num_nodes += 1
            if num_nodes == 0:
                continue
            adj.append([float(row[i]) for i in range(0, len(row))])

    adj = np.asarray(adj)
    return adj, num_nodes


def load_classes(path, type, max_labels=None):
    full_path = os.path.join(path, 'classes_{type}.txt'.format(type=type))
    classes = pd.read_csv(full_path)
    nans = pd.isna(classes['class_']).values
    classes.dropna(axis=0, inplace=True)
    classes['id'] = pd.factorize(classes.class_)[0]
    labels = classes['id'].values
    labels -= (np.min(labels) - 1)
    # labels = classes['id'].values.astype(int)
    print(labels)
    if (max_labels is None) or max_labels >= np.max(labels):
        num_classes = np.max(labels)
        num_graphs = labels.shape[0]
        labels -= np.ones(shape=(num_graphs,), dtype=int)
        one_hot_labels = np.zeros((num_graphs, num_classes))
        one_hot_labels[np.arange(num_graphs), labels] = 1
        return labels, one_hot_labels, num_graphs, num_classes, nans
    else:
        num_classes = max_labels
        num_graphs = labels.shape[0]
        for_one_hot = np.where(labels <= max_labels, labels, 0)
        labels = np.where(labels <= max_labels, labels, max_labels + 1)
        labels -= np.ones(shape=(num_graphs,), dtype=int)
        one_hot_labels = np.zeros((num_graphs, num_classes))
        one_hot_labels[np.arange(num_graphs), for_one_hot] = 1
        return labels, one_hot_labels, num_graphs, max_labels + 1, nans


def load_features(path, type, is_binary=False):
    full_path = os.path.join(path, 'data_{type}.txt'.format(type=type))
    num_nodes = -1
    features = []
    with open(full_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            num_nodes += 1
            if num_nodes == 0:
                continue
            if is_binary:
                features.append([1 if float(row[i]) > 0 else 0 for i in range(0, len(row))])
            else:
                features.append([float(row[i]) for i in range(0, len(row))])
    features = np.asarray(features)
    features = features.T
    return features




def gen_syn_data(
    n_classes=3,
    n_obs_train=200,
    n_obs_test=100,
    n_features=50,
    n_edges=3,
    n_characteristic_features=10,
    signal=[0.01, 0.01],
    diff_coef=[0.01, 0.01],
    noise=[0.01, 0.01],
    n_communities=5,
    probs=[0.5, 0.1],
    n_iter=3,
    model='SBM',
    random_seed=1996):
    """
    Generates a dataset. 
    Each class is defined by a set of characteristic features. 
    Each feature starts with random values. For each observation, the characteristic features of its class are increased by "signal".
    Then, values on each node diffuse through the graph.
    Parameters:
    ----------
    n_classes: number of classes
    n_obs: number of observations per class
    n_features: number of features, each corresponding to a node in the graph
    n_characteristic_features: number of features that are specific to each class
    signal: how much the value is increased for the characteristic features
    diff_coef: how much each value transmits its value to the next
    noise: noise level added at the end
    Returns:
    X: feature matrix
    y: labels
    graph: underlying graph
    """
    np.random.seed(random_seed)
    # Generate a scale-free graph with the Barabasi Albert model.
    if model=="BA":
        graph_train = ig.Graph.Barabasi(n_features, n_edges, directed=False)
        graph_test  = graph_train
    if model=='ER':
        graph_train = ig.Graph.Erdos_Renyi(n=n_features, m=n_edges*n_features, directed=False)
        graph_test  = graph_train
    if model=='SBM':
        n = [n_features // n_communities] * n_communities
        p = np.full((n_communities, n_communities), probs[1])
        adj_train = sbm(n=n, p=p)
        adj_test  = sbm(n=n, p=p)
        graph_train = ig.Graph.Adjacency(adj_train.tolist())
        graph_test  = ig.Graph.Adjacency(adj_test.tolist())
    X_train = []
    y_train = []
    X_test  = []
    y_test  = []
    for c in range(n_classes):
        # Draw the features which define this class
        characteristic_features = np.random.choice(n_features,size=n_characteristic_features,replace=False)
        for i in range(n_obs_train):
            # Start from a random vector
            features = np.abs(np.random.normal(0, 1, n_features))
            # TODO: force features to be positive or accept negative features ?
            # Increase the value for the characteristic features
            features[characteristic_features] += signal[0]
            features = features / np.linalg.norm(features)
            # Diffuse values through the graph
            # TODO: add different edge labels (positive or negative regulation)
            # TODO: maybe also give a weight to each edge
            for i in range(n_iter):
                features_next = np.copy(features)
                for e in graph_train.es:
                    features_next[e.target]+= (features[e.source] - features[e.target]) * diff_coef[0]
                    features_next[e.source]+= (features[e.target] - features[e.source]) * diff_coef[0]
                features = features_next
            if noise[0] > 0:
                features += np.random.normal(0, noise[0], n_features)
            X_train.append(features)
            y_train.append(c)

        for i in range(n_obs_test):
            # Start from a random vector
            features = np.abs(np.random.normal(0, 1, n_features))
            # TODO: force features to be positive or accept negative features ?
            # Increase the value for the characteristic features
            features[characteristic_features] += signal[1]
            features = features / np.linalg.norm(features)
            # Diffuse values through the graph
            # TODO: add different edge labels (positive or negative regulation)
            # TODO: maybe also give a weight to each edge
            for i in range(n_iter):
                features_next = np.copy(features)
                for e in graph_test.es:
                    features_next[e.target]+= (features[e.source] - features[e.target]) * diff_coef[1]
                    features_next[e.source]+= (features[e.target] - features[e.source]) * diff_coef[1]
                features = features_next
            if noise[1] > 0:
                features += np.random.normal(0, noise[1], n_features)
            X_test.append(features)
            y_test.append(c)
    train_idx = np.random.permutation(len(y_train)) - 1
    X_train   = np.array(X_train)[train_idx, :]
    y_train   = np.array(y_train)[train_idx]
    test_idx  = np.random.permutation(len(y_test)) - 1
    X_test    = np.array(X_test)[test_idx, :]
    y_test    = np.array(y_test)[test_idx]

    return X_train, np.array(y_train), graph_train.get_adjacency(), \
        np.array(X_test), np.array(y_test), graph_test.get_adjacency()