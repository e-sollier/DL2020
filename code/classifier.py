from model import *
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

class Classifier():
    def __init__(self,n_features,n_classes,n_layers, classifier='MLP', lr=.01, momentum=.9):
        if classifier == 'MLP': 
            self.net = MLP(n_features=n_features,n_layers=n_layers,n_classes=n_classes)
        if classifier == 'graphSAGE':
            self.net = GraphSage(n_features=n_features,n_layers=n_layers,n_classes=n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
    
    def fit(self,dataloader,epochs,verbose=False):
        self.net.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                self.optimizer.zero_grad()
                pred = self.net(batch)
                label = batch.y
                loss = self.criterion(pred,label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            total_loss /= len(dataloader.dataset)
            if verbose and epoch%(epochs//10)==0:
                print('[%d] loss: %.3f' % (epoch + 1,total_loss))
        

    def eval(self,dataloader,verbose=False):
        self.net.eval()
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in dataloader:
                X, graphs, labels = data
                y_true.append(labels)
                outputs = self.net(data)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.append(predicted)
        accuracy, conf_mat, precision, recall, f1_score = compute_metrics(y_true, y_pred)
        if verbose:
            print('Accuracy: {:.3f}'.format(accuracy))
            print('Confusion Matrix: \n', conf_mat)
            print('Precision: {:.3f}'.format(precision))
            print('Recall: {:.3f}'.format(recall))
            print('f1_score: {:.3f}'.format(f1_score))
        return accuracy, conf_mat, precision, recall, f1_score