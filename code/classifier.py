from model import *
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils import compute_metrics

class Classifier():
    def __init__(self,
        n_features,
        n_classes,
        n_hidden_GNN=[10],
        n_hidden_FC=[],
        K=4,
        dropout_GNN=0,
        dropout_FC=0, 
        classifier='MLP', 
        lr=.01, 
        momentum=.9,
        log_dir=None):
        if classifier == 'MLP': 
            self.net = NN(n_features=n_features, n_classes=n_classes,\
                n_hidden_FC=n_hidden_FC, dropout_FC=dropout_FC)
        if classifier == 'GraphSAGE':
            self.net = GraphSAGE(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN)
        if classifier == 'Chebnet':
            self.net = ChebNet(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN, K=K)
        if classifier == 'ConvNet':
            self.net = NNConvNet(n_features=n_features, n_classes=n_classes,\
                n_hidden_GNN=n_hidden_GNN, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_GNN=dropout_GNN)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.logging= log_dir is not None
        if self.logging:
            self.writer = SummaryWriter(log_dir=log_dir,flush_secs=1)
 
    def fit(self,data_loader,epochs,test_dataloader=None,verbose=False):
        if self.logging:
            data= next(iter(data_loader))
            self.writer.add_graph(self.net,[data.x,data.edge_index])

        
        for epoch in range(epochs):
            self.net.train()
            total_loss = 0
            
            for batch in data_loader:
                self.optimizer.zero_grad()
                pred = self.net(batch.x,batch.edge_index)
                label = batch.y
                loss = self.criterion(pred,label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs
            total_loss /= len(data_loader.dataset)
            if verbose and epoch%(epochs//10)==0:
                print('[%d] loss: %.3f' % (epoch + 1,total_loss))

            if self.logging:
                #Save the training loss, the training accuracy and the test accuracy for tensorboard vizualisation
                self.writer.add_scalar("Training Loss",total_loss,epoch)
                accuracy_train = self.eval(data_loader,verbose=False)[0]
                self.writer.add_scalar("Accuracy on Training Dataset",accuracy_train,epoch)
                if test_dataloader is not None:
                    accuracy_test = self.eval(test_dataloader,verbose=False)[0]
                    self.writer.add_scalar("Accuracy on Test Dataset",accuracy_test,epoch)
                


        

    def eval(self,data_loader,verbose=False):
        self.net.eval()
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in data_loader:
                X, graphs, labels = data.x, data.edge_index, data.y
                y_true.extend(list(labels))
                outputs = self.net(data.x,data.edge_index)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(list(predicted))
        accuracy, conf_mat, precision, recall, f1_score = compute_metrics(y_true, y_pred)
        if verbose:
            print('Accuracy: {:.3f}'.format(accuracy))
            print('Confusion Matrix:n', conf_mat)
            print('Precision: {:.3f}'.format(precision))
            print('Recall: {:.3f}'.format(recall))
            print('f1_score: {:.3f}'.format(f1_score))
        return accuracy, conf_mat, precision, recall, f1_score