from model import *
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

class Classifier():
    def __init__(self,n_features,n_classes,n_layers, classifier='MLP'):
        if classifier == 'MLP': 
            self.net = MLP(n_features=n_features,n_layers=n_layers,n_classes=n_classes)
        if classifier == 'graphSAGE':
            self.net = GraphSage(n_features=n_features,n_layers=n_layers,n_classes=n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
    
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
        with torch.no_grad():
            for data in dataloader:
                X, labels = data
                outputs = self.net(X)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if verbose:
            print('Accuracy of the network: %d %%' % (100 * correct / total))
        return (100 * correct / total)