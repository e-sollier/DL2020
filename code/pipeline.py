import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
import torch_geometric.nn as pyg_nn


class Pipeline():
    def __init__(
        self, 
        tag, 
        type='synthetic',         
        n_features=200,
        n_obs_min=100,
        n_obs_max=500,
        raw_dir='data_raw', 
        input_dir='data_input', 
        output_dir='output', 
        models_dir='models', 
        log_dir='.logs',
        io_type='create', #or 'load'
        seed=1375,
        verbose=True,
        log=True
        ):

        #self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

    def create_dataset(self, **kwargs):
        pass

    def load_dataset(self, tag):
        pass

    def create_model(self,
        type='GCNConv',
        epochs=20, #TODO: early stopping
        lr=.01,
        dropout=.01,
        optimizer='',
        criterion='',
        **kwargs
        ):
        pass

    def train(self):
        pass

    def test(self):
        pass




