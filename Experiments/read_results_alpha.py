import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#directory containing the outputs of run.py (json files)
results_dir = "Experiments/out_alpha"
l=[]
for f in os.listdir(results_dir):
    with open(os.path.join(results_dir,f)) as json_file:
        data = json.load(json_file)
        if float(data["alpha"])<20:
            l.append(data)
df = pd.DataFrame(l)

x_var = "alpha" # name of the variable that will be plotted on the x-axis
y_var = "accuracy" # name of the variable that will be plotted on the y-axis

type_vars = ["n_obs_train"]

res={}
for x in df.index:
    type_name = "_".join([str(df.loc[x,t]) for t in type_vars])
    if not type_name in res:
        res[type_name] = {}
    x_value = df.loc[x,x_var]
    if not x_value in res[type_name]:
        res[type_name][x_value]=[]
    res[type_name][x_value].append(df.loc[x,y_var])

plt.rcParams.update({'font.size': 22})
keys_sorted = [str(x) for x in sorted([int(x) for x in res])]
for t in keys_sorted:
    x=[]
    y=[]
    for x_val in res[t]:
        x.append(float(x_val))
        y.append(np.mean(res[t][x_val]))
    x = np.array(x)
    y = np.array(y)
    #reorder according to x value.
    ind = np.argsort(x)
    x=x[ind]
    y=y[ind]
    plt.plot(x,y,label=t + " training obs. per class")



plt.xlabel(x_var)
plt.ylabel(y_var)
plt.legend()
plt.show()