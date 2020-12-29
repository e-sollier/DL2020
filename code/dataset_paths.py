import pandas as pd
# from utils import *
from dataset import *
# Dataset


input_dir  = 'data_input'
output_dir = 'data_input'

params_dt = pd.DataFrame({
    'n_classes' : [[5]],
    'n_obs_train' : [[500]],
    'n_obs_test' : [[200]],
    'n_features' : [[50, 100, 250]],
    'n_char_features' : [[0.1, 0.25, 0.5]],
    # 'n_edges' : [[500, 1000]],
    'signal' : [[[10, 1], [10, 2.5], [10, 5], [10, 10]]],
    'diff_coef' : [[[0.5, 0.5]]],
    'noise' : [[[.5, .5]]],
    'n_communities' : [[5]],
    'probs' : [[[0.9, 0.1]]],
    'n_iter' : [[1, 2]],
    'model' : [['SBM']],
    'syn_method' : [['diffusion', 'oppneighbors', 'activation']],
    'est_method' : [['glasso']],
    'alphas'     : [[0.02, 0.05]]})

params_dt['tag'] = (range(params_dt.shape[0]) + 1)
df['tag'] = df['tag'].apply(lambda x: f"EXP{x}")

for column in params_dt:
    params_dt = params_dt.explode(column)

def create_EXP(params, save=False):
    dataset = Dataset(tag=params['tag'], input_dir= 'data_input', output_dir='data_input')
    dataset.create_syn(
        n_classes=params['n_classes'], 
        n_obs_train=params['n_obs_train'], 
        n_obs_test=params['n_obs_test'], 
        n_features=params['n_features'],
        n_char_features=params['n_char_features'], 
        signal=params['signal'], 
        diff_coef=params['diff_coef'], 
        noise=params['noise'], 
        n_communities=params['n_communities'],
        probs=params['probs'], 
        n_iter=params['n_iter'], 
        model=params['model'],
        syn_method=params['syn_method'])
    dataset.create_graph(alphas=params['alphas'])
    train_dataloader = dataset._dataloader('train', use_true_graph=False, batch_size=32)
    test_dataloader  = dataset._dataloader('test', use_true_graph=False, batch_size=32)
    if save:
        dataset.save()
    return train_dataloader, test_dataloader

params_dt[1:5].apply(create_EXP, axis=1)


