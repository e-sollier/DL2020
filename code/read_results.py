import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#directory containing the outputs of run.py (json files)
results_dir = "code/results"
l=[]
for f in os.listdir(results_dir):
    with open(os.path.join(results_dir,f)) as json_file:
        data = json.load(json_file)
        l.append(data)
df = pd.DataFrame(l)

x_var = "n_obs_train" # name of the variable that will be plotted on the x-axis
y_var = "accuracy" # name of the variable that will be plotted on the y-axis

type_vars = ["classifier","n_hidden_GNN","n_hidden_FC"] # names of the variables which define one classifier type

res={}
for x in df.index:
    type_name = "_".join([str(df.loc[x,t]) for t in type_vars])
    if not type_name in res:
        res[type_name] = {"x":[],"y":[]}
    res[type_name]["x"].append(df.loc[x,x_var])
    res[type_name]["y"].append(df.loc[x,y_var])

for t in res:
    x = np.array(res[t]["x"])
    y = np.array(res[t]["y"])
    #reorder according to x value.
    ind = np.argsort(x)
    x=x[ind]
    y=y[ind]
    plt.plot(x,y,label=t)

plt.xlabel(x_var)
plt.ylabel(y_var)
plt.legend()
plt.show()