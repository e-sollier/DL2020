import argparse
import json
import os

import torch
from classifier import Classifier
from dataset import Dataset
from hyperparameters import get_hyperparams


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str,required=True, help='Input dataset. Can either be the name of the synthetic dataset generation method, \
                                              or the path to a real dataset.')
parser.add_argument('-o', type=str,required=True, help='Output directory.')
parser.add_argument('--n_classes',type=int,default=3,help='Number of classes')
parser.add_argument('--n_features',type=int,default=100,help='Number of features (in case of synthetic data')
parser.add_argument('--n_char_features',type=int,default=10,help='Number of characteristic features for each class')
parser.add_argument('--n_obs_train',type=int,default=500,help='Number of observations for training (per class)')
parser.add_argument('--n_obs_test',type=int,default=5000,help='Number of observations for testing (per class)')
parser.add_argument('--graph_model',type=str,default='BA',help='Graph generation model')
parser.add_argument('--signal_train',type=float,default=0,help='Signal for training data')
parser.add_argument('--signal_test',type=float,default=0,help='Signal for test data')
parser.add_argument('--diff_train',type=float,default=0.3,help='Diffusion coefficient for training data')
parser.add_argument('--diff_test',type=float,default=0.3,help='Diffusion coefficient for test data')
parser.add_argument('--noise_train',type=float,default=0.2,help='Noise for training data')
parser.add_argument('--noise_test',type=float,default=0.2,help='Noise for test data')
parser.add_argument('--n_iter',type=int,default=1,help='Number of diffusion iterations')
parser.add_argument('--n_communities',type=int,default=5,help='Number of communities (for SBM)')
parser.add_argument('--probs_train',type=float,default=0.9,help='SBM probabilities for training')
parser.add_argument('--probs_test',type=float,default=0.1,help='SBM probabilities for testing')

parser.add_argument('--classifier',type=str,default="MLP",help='Type of classifier')
parser.add_argument('--n_hidden_GNN',type=int,default=0,help='Number of hidden features for the GNN. If 0, do not use a GNN.')
parser.add_argument('--n_hidden_FC',type=int,default=0,help='Number of features in the fully connected hidden layer. If 0, do not use a hidden layer.')
parser.add_argument('--n_hidden_FC2',type=int,default=0,help='Number of features in the 2nd fully connected hidden layer. If 0, do not use a 2nd hidden layer.')
parser.add_argument('--n_hidden_FC3',type=int,default=0,help='Number of features in the 3rd fully connected hidden layer. If 0, do not use a 3rd hidden layer.')
parser.add_argument('--K',type=int,default=2,help='Parameter for Cheb GNN.')
parser.add_argument('--epochs',type=int,default=30,help='Number of training epochs.')
parser.add_argument('--batch_size',type=int,default=16,help='Batch size for the neural networks.')

parser.add_argument('--infer_graph',type=str,default="False",help="Whether to infer the graph from the data (True) or directly use the true graph.")
parser.add_argument('--FPR',type=float,default=0.0,help="Percentage of false edges in the noisy graph.")
parser.add_argument('--FNR',type=float,default=0.0,help="Percentage of true edges which are removed in the noisy graph.")
parser.add_argument('--alpha',type=float,help='Parameter for graph inferrence.')
parser.add_argument('--CV_alpha',type=str,default='False',help='Whether or not to run CV for the dropout parameter.')
parser.add_argument('--dropout',type=float,default=0.2,help='Dropout rate.')
parser.add_argument('--CV_dropout',type=str,default='False',help='Whether or not to run CV for the dropout parameter.')

parser.add_argument('--seed',type=int,default=0,help='Seed for generating the synthetic dataset.')




args = parser.parse_args()
n_hidden_GNN = [] if args.n_hidden_GNN==0 else [args.n_hidden_GNN]

if args.n_hidden_FC==0:
  n_hidden_FC=[]
else:
  if args.n_hidden_FC2==0:
    n_hidden_FC = [args.n_hidden_FC]
  else:
    if args.n_hidden_FC3==0:
      n_hidden_FC = [args.n_hidden_FC,args.n_hidden_FC2]
    else:
      n_hidden_FC = [args.n_hidden_FC,args.n_hidden_FC2,args.n_hidden_FC3]

use_true_graph = args.infer_graph=="False" and args.FPR==0.0 and args.FNR==0.0
infer_graph = args.infer_graph=="True"

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)

# Create a synthetic dataset or load a real dataset
dataset = Dataset(tag='EXP1',random_seed=args.seed)
if args.i in ["diffusion","activation","sign"]:
    dataset.create_syn(n_classes = args.n_classes, 
                    n_obs_train = args.n_obs_train, 
                    n_obs_test= args.n_obs_test, 
                    n_features=args.n_features,
                    n_char_features = args.n_char_features, 
                    signal =[args.signal_train, args.signal_test], 
                    diff_coef=[args.diff_train, args.diff_test], 
                    noise = [args.noise_train, args.noise_test], 
                    n_communities = args.n_communities,
                    probs = [args.probs_train, args.probs_test], 
                    n_iter=args.n_iter, 
                    model =args.graph_model,
                    syn_method=args.i)
else:
    #TODO: load real dataset
    raise("unimplemented.")

# Select the dropout rate and alpha
CV_alpha = args.CV_alpha=="True"
CV_dropout = args.CV_dropout=="True"
dropout_rate,alpha = get_hyperparams(CV_dropout=CV_dropout,CV_alpha=CV_alpha,dataset=dataset,n_features=args.n_features, n_obs_train=args.n_obs_train,\
        n_classes=args.n_classes,n_hidden_GNN=n_hidden_GNN,n_hidden_FC=n_hidden_FC,K=args.K,classifier=args.classifier,\
        lr=0.001,momentum=0.9,epochs=args.epochs,device=device,batch_size=args.batch_size,use_true_graph=use_true_graph,dropout_rate=args.dropout,alpha=args.alpha)


"""
# Select the dropout rate and alpha
def select_alpha(n_obs):
  if n_obs<=250:
    return 0.5
  elif n_obs<=600:
    return 1
  elif n_obs <=1200:
    return 2
  elif n_obs<=1700:
    return 2.5
  elif n_obs <=2000:
    return 3
  elif n_obs <=3500:
    return 4
  elif n_obs <=5500:
    return 5
  elif n_obs <= 6500:
    return 6
  elif n_obs <=7500:
    return 8
  elif n_obs <= 9000:
    return 11
  else:
    return 15

CV_alpha = args.CV_alpha=="True"
CV_dropout = args.CV_dropout=="True"
if CV_alpha or CV_dropout: # run CV
  if CV_dropout:
    dropout_rate_list=[0,0.1,0.2,0.5] # grid of dropout rate values
  else:
    dropout_rate_list=args.dropout
  if CV_alpha:
    alpha_list = [0.5,1,2,3,4] # grid of alpha values
  else:
    if args.alpha is None:
      #select alpha based on the number of observations
      alpha_list = [select_alpha(args.n_obs_train)]
    else:
      alpha_list=[args.alpha]
  dropout_rate,alpha = select_hyperparameters_CV(dataset=dataset,n_features=args.n_features,n_classes=args.n_classes,n_hidden_GNN=n_hidden_GNN,n_hidden_FC=n_hidden_FC,\
        K=args.K,classifier=args.classifier,lr=0.001,momentum=0.9,epochs=args.epochs,device=device,batch_size=args.batch_size,dropout_rate_list=dropout_rate_list,\
         alpha_list=alpha_list,use_true_graph=use_true_graph,graph_method="glasso_R")
  print("Selected dropout rate: " + str(dropout_rate))
  print("Selected alpha: " + str(alpha))
else: # don't run CV
  dropout_rate = args.dropout
  if args.alpha is None:
    alpha = select_alpha(args.n_obs_train)
  else:
    alpha = args.alpha"""
  



# Create dataloader
if infer_graph: # Infer the graph with glasso
  dataset.create_graph(alphas=alpha)
else: # use a noisy version of the true graph
  dataset.create_noisy_true_graph(FPR=args.FPR,FNR=args.FNR)
train_dataloader = dataset._dataloader('train',use_true_graph=use_true_graph,batch_size=args.batch_size)
test_dataloader  = dataset._dataloader('test',use_true_graph=use_true_graph,batch_size=args.batch_size)

# Fit and evaluate a classifier
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

clf.fit(train_dataloader, epochs = args.epochs, test_dataloader=test_dataloader,verbose=True)

results = clf.eval(test_dataloader, verbose=False)


# Store the results in a json file.
output = {"accuracy":results[0],"precision":results[2],"recall":results[3],"f1":results[4], "n_classes":args.n_classes,"n_features":args.n_features,\
      "n_char_features":args.n_char_features,"n_obs_train":args.n_obs_train,"n_obs_test":args.n_obs_test,\
      "signal_train":args.signal_train,"signal_test":args.signal_test,"diff_train":args.diff_train,"diff_test":args.diff_test,"noise_train":args.noise_train,\
        "noise_test":args.noise_test,\
        "classifier":args.classifier,"n_hidden_GNN":args.n_hidden_GNN,"n_hidden_FC":args.n_hidden_FC,"n_hidden_FC2":args.n_hidden_FC2,"n_hidden_FC3":args.n_hidden_FC3,\
          "seed":args.seed,"alpha":str(args.alpha),\
          "FPR":args.FPR,"FNR":args.FNR,"infer_graph":args.infer_graph}

filename = "_".join([args.i,str(args.n_features),str(args.n_classes),str(args.n_char_features),str(args.n_obs_train),str(args.n_obs_test),args.graph_model,\
    str(args.signal_train),str(args.signal_test),str(args.diff_train),str(args.diff_test),str(args.noise_train),str(args.noise_test),\
      args.classifier,str(args.n_hidden_GNN),str(args.n_hidden_FC),str(args.n_hidden_FC2),str(args.n_hidden_FC3),str(args.seed),str(args.alpha),\
        str(args.FNR),str(args.FPR),str(args.infer_graph)]) + ".json"
if not os.path.exists(args.o):
    os.makedirs(args.o)
with open(os.path.join(args.o,filename), 'w') as outfile:
    json.dump(output, outfile)
