import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

#directory containing the outputs of run.py (json files)
results_dir = "output/results_2020_12_30"
n_obs_test  = 500
l=[]
for f in os.listdir(results_dir):
    with open(os.path.join(results_dir,f)) as json_file:
        data = json.load(json_file)
        l.append(data)
df = pd.DataFrame(l)

index_cols_mean = ['tag', 'classifier', 'n_obs_train', 'n_hidden_GNN', \
    'n_hidden_FC', 'n_hidden_FC2', 'n_hidden_FC3', 'alpha', 'K']
index_cols_max  = ['tag', 'classifier', 'n_obs_train']
df = df.loc[df['n_obs_test'] == n_obs_test] 
df = df.loc[(df['n_hidden_FC'] >= df['n_hidden_FC2']) & (df['n_hidden_FC2'] >= df['n_hidden_FC3'])]
df = df.loc[df['n_hidden_FC2'] == 0]
# dfs = [df.loc[df['tag']== t] for t in set(df['tag'])]
# fig, axis = plt.subplots(1, len(set(df['tag'])))
# for i in range(len(dfs)):
#     dt = dfs[i].groupby(index_cols_mean).agg({'accuracy': ['mean']})
#     dt.columns = ['accuracy']
#     dt = dt.reset_index()
#     dt = dt.groupby(index_cols_max).agg({'accuracy': ['max']})
#     dt.columns = ['accuracy']
#     dt = dt.reset_index()
#     dt = dt.pivot_table(index=['n_obs_train'], \
#         columns='classifier', values='accuracy')
#     ax = dt.plot(ax=axis[i], title=list(set(df['tag']))[i])
#     ax.set_xscale('log')
#     ax.set_xlabel('Number of training observations per class (log scale)')
#     ax.set_ylabel('Accuracy')
#     handles, labels = ax.get_legend_handles_labels()
    # ax.get_legend().remove()


df = df.loc[df['tag'] == 'pbmc']
dt = df.groupby(index_cols_mean).agg({'accuracy': ['mean']})
dt.columns = ['accuracy']
dt = dt.reset_index()
dt = dt.groupby(index_cols_max).agg({'accuracy': ['max']})
dt.columns = ['accuracy']
dt = dt.reset_index()
dt = dt.pivot_table(index=['n_obs_train'], \
    columns='classifier', values='accuracy')
ax = dt.plot()
ax.set_xscale('log')
ax.set_xlabel('Number of training observations per class (log scale)')
ax.set_ylabel('Accuracy')
ax.set_xticks([])
ax.set_xticks([125, 250, 500, 1000], minor=True)
ax.axes.xaxis.set_ticklabels([125, 250, 500, 1000])
# handles, labels = ax.get_legend_handles_labels()

# plt.legend(handles, labels, loc='center right')
# plt.show()
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.savefig('output/figs/n_obs_real.png', dpi=1000)
