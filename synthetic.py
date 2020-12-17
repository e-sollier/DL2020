import numpy as np
import igraph as ig 


def generate_dataset(nb_classes,nb_obs,nb_features,nb_edges,nb_characteristic_features,signal,diffusion_coefficient,noise,random_seed):
    """
    Generates a dataset. 
    Each class is defined by a set of characteristic features. 
    Each feature starts with random values. For each observation, the characteristic features of its class are increased by "signal".
    Then, values on each node diffuse through the graph.
    Parameters:
    ----------
    nb_classes: number of classes
    nb_obs: number of observations per class
    nb_features: number of features, each corresponding to a node in the graph
    nb_characteristic_features: number of features that are specific to each class
    signal: how much the value is increased for the characteristic features
    diffusion_coefficient: how much each value transmits its value to the next
    noise: noise level
    Returns:
    X: feature matrix
    y: labels
    graph: underlying graph
    """
    np.random.seed(random_seed)
    # Generate a scale-free graph with the Barabasi Albert model.
    graph = ig.Graph.Barabasi(nb_features,nb_edges,directed=False)
    X = []
    y= []
    for c in range(nb_classes):
        # Draw the features which define this class
        characteristic_features = np.random.choice(nb_features,size=nb_characteristic_features,replace=False)
        for i in range(nb_obs):
            # Start from a random vector
            features = np.random.normal(0,1,nb_features)
            # TODO: force features to be positive or accept negative features ?
            # Increase the value for the characteristic features
            features[characteristic_features] += signal
            # Diffuse values through the graph
            # TODO: add different edge labels (positive or negative regulation)
            # TODO: maybe also give a weight to each edge
            # TODO: maybe do several iterations of diffusion ?  
            features_next = np.copy(features)
            for e in graph.es:
                features_next[e.target]+= features[e.source] * diffusion_coefficient
            # TODO: add additional noise on top ? 
            X.append(features_next)
            y.append(c)
    return X,y,graph



