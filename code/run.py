import argparse
import json
import os

import torch
from classifier import Classifier
from dataset import Dataset
from hyperparameters import select_hyperparameters_CV


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str,required=True, help='Input dataset. Can either be the name of the synthetic dataset generation method, \
                                              or the path to a real dataset.')
parser.add_argument('-o', type=str,required=True, help='Output directory.')
parser.add_argument('--n_classes',type=int,default=5,help='Number of classes')
parser.add_argument('--n_features',type=int,default=50,help='Number of features (in case of synthetic data')
parser.add_argument('--n_char_features',type=int,default=6,help='Number of characteristic features for each class')
parser.add_argument('--n_obs_train',type=int,default=500,help='Number of observations for training (per class)')
parser.add_argument('--n_obs_test',type=int,default=1000,help='Number of observations for testing (per class)')
parser.add_argument('--graph_model',type=str,default='ER',help='Graph generation model')
parser.add_argument('--signal_train',type=float,default=2,help='Signal for training data')
parser.add_argument('--signal_test',type=float,default=2,help='Signal for test data')
parser.add_argument('--diff_train',type=float,default=0.3,help='Diffusion coefficient for training data')
parser.add_argument('--diff_test',type=float,default=0.3,help='Diffusion coefficient for test data')
parser.add_argument('--noise_train',type=float,default=0.4,help='Noise for training data')
parser.add_argument('--noise_test',type=float,default=0.4,help='Noise for test data')
parser.add_argument('--n_iter',type=int,default=1,help='Number of diffusion iterations')
parser.add_argument('--n_communities',type=int,default=5,help='Number of communities (for SBM)')
parser.add_argument('--probs_train',type=float,default=0.9,help='SBM probabilities for training')
parser.add_argument('--probs_test',type=float,default=0.1,help='SBM probabilities for testing')

parser.add_argument('--classifier',type=str,default="MLP",help='Type of classifier')
parser.add_argument('--n_hidden_GNN',type=int,default=0,help='Number of hidden features for the GNN. If 0, do not use a GNN.')
parser.add_argument('--n_hidden_FC',type=int,default=0,help='Number of features in the fully connected hidden layer. If 0, do not use a hidden layer.')
parser.add_argument('--K',type=int,default=4,help='Parameter for Cheb GNN.')



args = parser.parse_args()
n_hidden_GNN = [] if args.n_hidden_GNN==0 else [args.n_hidden_GNN]
n_hidden_FC = [] if args.n_hidden_FC==0 else [args.n_hidden_FC]

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)


dataset = Dataset(tag='EXP1')

if args.i in ["diffusion","oppneighbors"]:
    dataset.create_syn(n_classes = args.n_classes, 
                    n_obs_train = args.n_obs_train, 
                    n_obs_test= args.n_obs_test, 
                    n_features=args.n_features,
                    n_char_features = args.n_char_features, 
                    signal =[args.signal_train, args.signal_test], 
                    diff_coef=[args.diff_train, args.diff_test], 
                    noise = [args.noise_train, .4], 
                    n_communities = args.n_communities,
                    probs = [args.probs_train, args.probs_test], 
                    n_iter=args.n_iter, 
                    model =args.graph_model,
                    syn_method=args.i)
else:
    #TODO: load real dataset
    raise("unimplemented.")

dataset.create_graph(alphas=[0.001, 0.002])

train_dataloader = dataset._dataloader('train',use_true_graph=True,batch_size=16)
test_dataloader  = dataset._dataloader('test',use_true_graph=True,batch_size=16)

#dropout_rate = 0.1 
dropout_rate = select_hyperparameters_CV(dataset=dataset,n_features=args.n_features,n_classes=args.n_classes,n_hidden_GNN=n_hidden_GNN,n_hidden_FC=n_hidden_FC,\
        K=args.K,classifier=args.classifier,lr=0.001,momentum=0.9,epochs=30,device=device,batch_size=16)
print("Selected dropout rate: " + str(dropout_rate))


clf = Classifier(n_features=args.n_features,
        n_classes=args.n_classes,
        n_hidden_GNN=n_hidden_GNN,
        n_hidden_FC=n_hidden_FC,
        dropout_GNN=dropout_rate, 
        dropout_FC=dropout_rate,
        K=args.K,
        classifier=args.classifier, 
        lr=.001, 
        momentum=.9,
        log_dir=None)

clf.fit(train_dataloader, epochs = 30, test_dataloader=test_dataloader,verbose=True)

results = clf.eval(test_dataloader, verbose=False)

output = {"accuracy":results[0],"precision":results[2],"recall":results[3],"f1":results[4]}

filename = "_".join([args.i,str(args.n_features),str(args.n_classes),str(args.n_char_features),str(args.n_obs_train),str(args.n_obs_test),args.graph_model,\
    str(args.signal_train),str(args.signal_test),str(args.diff_train),str(args.diff_test),str(args.noise_train),str(args.noise_test),\
      args.classifier,str(args.n_hidden_GNN),str(args.n_hidden_FC)]) + ".json"
if not os.path.exists(args.o):
    os.makedirs(args.o)
with open(os.path.join(args.o,filename), 'w') as outfile:
    json.dump(output, outfile)