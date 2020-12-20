import numpy as np
import igraph as ig
import torch
from torch_geometric.data import Data, DataLoader



def get_dataloader(graph, X, y, undirected=False):
    """
    Converts an igraph graph and a dataset (X,y) to a dataloader of pytorch geometrics graphs.
    In the output, all of the graphs will have the same connectivity (given by the input graph),
    but the node features will be the features from X.
    """
    n_obs,n_features = X.shape

    # Convert the igraph graph to a pytorch edge_index
    sources=[]
    targets=[]
    for edge in graph.es:
        sources.append(edge.source)
        targets.append(edge.target)
        if undirected:
            sources.append(edge.target)
            targets.append(edge.source)
    edge_index = torch.tensor([sources,targets])

    # Create the graph for each observation. 
    # Each observarion has the same graph structure, but diffrerent node_features (x).
    list_graphs = []
    for i in range(n_obs):
        y_tensor = torch.tensor(y[i])
        x_tensor = torch.tensor(X[i,:]).view(X.shape[1],1).float() # X.shape[1] nodes and 1 feature per node
        data = Data(x = x_tensor, edge_index = edge_index, y= y_tensor )
        #data.num_nodes = X.shape[1]

        list_graphs.append(data.coalesce())


    
    dataloader = DataLoader(list_graphs,batch_size=32,shuffle=True)
    return dataloader