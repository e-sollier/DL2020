import numpy as np
from synthetic import generate_dataset
from data_to_adjacency import get_adjacency_matrix_GL
from adjacency_score import performance_metrics

# Simulation of Experiments with different number of features, observations, nodes;
# number of classes, characteristic features; signal-to-noise ratio.

# Output: score, i.e. the measure of accuracy for estimated adjacency matrix vs. the true one.

def simulation_and_estimation(nb_classes,nb_obs,nb_features,nb_edges,nb_characteristic_features,signal,diffusion_coefficient,noise,nb_simulations):
    
    TP = []
    TN = []
    FP = []
    FN = []
    precision = []
    recall = []
    f1_score = []
    
    for s in range(nb_simulations):
    
        X_s,y_s,graph_s = generate_dataset(nb_classes,nb_obs,nb_features,nb_edges,\
                                           nb_characteristic_features,signal,diffusion_coefficient,noise,random_seed=s)

        true_adj_s = np.array(graph_s.get_adjacency().data)
    
        est_adj_s = get_adjacency_matrix_GL(X_s)
    
        TP_s, TN_s, FP_s, FN_s, precision_s, recall_s, f1_score_s = performance_metrics(true_adj_s, est_adj_s)
    
        TP.append(TP_s)
        TN.append(TN_s)
        FP.append(FP_s)
        FN.append(FN_s)
        precision.append(precision_s)
        recall.append(recall_s)
        f1_score.append(f1_score_s)
   
    
    avg_TP = np.mean(TP)
    avg_TN = np.mean(TN)
    avg_FP = np.mean(FP)
    avg_FN = np.mean(FN)
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1_score = np.mean(f1_score)
    
    return avg_TP, avg_TN, avg_FP, avg_FN, avg_precision, avg_recall, avg_f1_score




