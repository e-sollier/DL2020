import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
import torch_geometric.nn as pyg_nn



class NN(nn.Module):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[], \
        n_hidden_FC=[10], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(NN, self).__init__()
        self.FC           = True
        self.n_features   = n_features
        self.n_classes    = n_classes
        self.layers_GNN   = nn.ModuleList()
        self.layers_FC    = nn.ModuleList()
        self.n_layers_GNN = len(n_hidden_GNN)
        self.n_layers_FC  = len(n_hidden_FC)
        self.dropout_GNN  = dropout_GNN
        self.dropout_FC   = dropout_FC
        self.n_hidden_GNN = n_hidden_GNN
        self.n_hidden_FC  = n_hidden_FC

        if self.n_layers_GNN > 0:
            self.FC = False
        print(self.FC)

        if self.n_layers_FC > 0:
            self.layers_FC.append(nn.Linear(n_features, n_hidden_FC[0]))
            if self.n_layers_FC > 1:
                print((self.n_layers_FC-2))
                for i in range(self.n_layers_FC-1):
                    print(i)
                    self.layers_FC.append(nn.Linear(n_hidden_FC[i], n_hidden_FC[(i+1)]))
            self.last_layer_FC = nn.Linear(n_hidden_FC[(self.n_layers_FC-1)], n_classes)
        else:
            self.last_layer_FC = nn.Linear(n_features, n_classes)

    def forward(self,x,edge_index):
        if self.FC == True:
            # Resize from (1,batch_size * n_features) to (batch_size, n_features)
            x = x.view(-1,self.n_features)
        for layer in self.layers_GNN:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout_GNN, training=self.training)
        if self.n_layers_GNN > 0:
            x = x.view(-1, self.n_features*self.n_hidden_GNN[(self.n_layers_GNN-1)])
            x = F.relu(self.last_layer_GNN(x))
            x = F.dropout(x, p=self.dropout_GNN, training=self.training)
        for layer in self.layers_FC:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout_FC, training=self.training)
        x = self.last_layer_FC(x)
        return x


class GraphSAGE(NN):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[10], \
        n_hidden_FC=[], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(GraphSAGE, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.SAGEConv(1, n_hidden_GNN[0]))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.SAGEConv(n_hidden_GNN[i], n_hidden_GNN[(i+1)]))
        self.last_layer_GNN = \
            nn.Linear(n_features*n_hidden_GNN[(self.n_layers_GNN-1)], n_features)


class ChebNet(NN):
    def __init__(self,
        n_features,
        n_classes,
        n_hidden_GNN=[10],
        n_hidden_FC=[],
        K=4,
        dropout_GNN=0,
        dropout_FC=0):
        super(ChebNet, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.ChebConv(1, n_hidden_GNN[0], K))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.ChebConv(n_hidden_GNN[i], n_hidden_GNN[(i+1), K]))
        self.last_layer_GNN = \
            nn.Linear(n_features*n_hidden_GNN[(self.n_layers_GNN-1)], n_features)


class NNConvNet(NN):
    def __init__(self, \
        n_features, \
        n_classes, \
        n_hidden_GNN=[10], \
        n_hidden_FC=[], \
        dropout_GNN=0, \
        dropout_FC=0):
        super(NNConvNet, self).__init__(\
            n_features, n_classes, n_hidden_GNN,\
            n_hidden_FC, dropout_FC, dropout_GNN)

        self.layers_GNN.append(pyg_nn.NNConv(1, n_hidden_GNN[0]))
        if self.n_layers_GNN > 1:
            for i in range(self.n_layers_GNN-1):
                self.layers_GNN.append(pyg_nn.NNConv(n_hidden_GNN[i], n_hidden_GNN[(i+1)]))
        self.last_layer_GNN = \
            nn.Linear(n_features*n_hidden_GNN[(self.n_layers_GNN-1)], n_features)