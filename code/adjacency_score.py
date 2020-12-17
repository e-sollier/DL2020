import numpy as np

def performance_metrics(data_true,data_est):

    TP = np.sum(data_true[data_true==1] == data_est[data_true==1]) # true positive rate
    TN = np.sum(data_true[data_true==0] == data_est[data_true==0]) # true negative rate
    FP = np.sum(data_true[data_true==0] != data_est[data_true==0]) # false positive rate
    FN = np.sum(data_true[data_true==1] != data_est[data_true==1]) # false negative rate
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2*precision*recall/(precision+recall)
    
    return TP, TN, FP, FN, precision, recall, f1_score

########## One can use also alternatively the following ##################3

# from sklearn.metrics import precision_recall_fscore_support
# ps,rs,fs,ss = precision_recall_fscore_support(true_adj, adj_v2, average = 'micro')

###### This produces the same result as the function above ############