from model import *
import torch.optim as optim
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
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            if verbose and epoch%(epochs//10)==0:
                print('[%d] loss: %.3f' % (epoch + 1, running_loss/len(dataloader.dataset)))

    def eval(self,dataloader,verbose=False):
        self.net.eval()
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in dataloader:
                X, labels = data
                y_true.append(labels)
                outputs = self.net(X)
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