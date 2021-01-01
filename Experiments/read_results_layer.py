import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#directory containing the outputs of run.py (json files)
results_dir = "Experiments/out_layer"
l=[]
for f in os.listdir(results_dir):
    with open(os.path.join(results_dir,f)) as json_file:
        data = json.load(json_file)
        l.append(data)
df = pd.DataFrame(l)

x_var = "n_obs_train" # name of the variable that will be plotted on the x-axis
y_var = "accuracy" # name of the variable that will be plotted on the y-axis

type_vars = ["classifier"]

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
for t in res:
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
    plt.plot(x,y,label=t,linewidth=4)



plt.xlabel("Number of training observations per class (log scale)")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.legend()
plt.show()