import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
import torch_geometric.nn as pyg_nn


class GNN_classifier():
    def __init__(self,n_features,n_classes):
        self.net = GNN(n_features=n_features,n_classes=n_classes,n_hidden=4)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        #self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

    def fit(self,dataloader,epochs,verbose=False):
        self.net.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader.dataset:
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
        for data in dataloader.dataset:
            with torch.no_grad():
                pred = self.net(data)
                pred = pred.argmax(dim=1)
                label = data.y
                
            correct += pred.eq(label).sum().item()
            total = len(dataloader.dataset) 
        
        if verbose:
            print('Accuracy of the network: %d %%' % (100 * correct / total))
        return (100 * correct / total)

class GNN(nn.Module):
    def __init__(self,n_features,n_classes,n_hidden=4):
        super(GNN, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        #self.conv1 = GCNConv(1, n_hidden)
        #self.conv1 = pyg_nn.GINConv(nn.Sequential(nn.Linear(1,n_hidden),nn.ReLU(), nn.Linear(n_hidden, n_hidden)))

        #Use GraphSAGE layer:
        #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
        self.conv1 = pyg_nn.SAGEConv(1,n_hidden)
        #self.conv2 = GCNConv(n_hidden,n_hidden)
        self.linear = nn.Linear(n_features*n_hidden,n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        
        # Resize from (batch_size * n_features, n_hidden) to (batch_size, n_features * n_hidden)
        x = x.view(-1,self.n_features*self.n_hidden)
        x = self.linear(x)

        return x
        #return F.log_softmax(x, dim=0)



