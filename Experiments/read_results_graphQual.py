import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#directory containing the outputs of run.py (json files)
results_dir = "Experiments/out_graphQual"
results_dirMLP = "Experiments/out_graphQualMLP"
l=[]
for f in os.listdir(results_dir):
    with open(os.path.join(results_dir,f)) as json_file:
        data = json.load(json_file)
        l.append(data)
df = pd.DataFrame(l)
var1 = "FPR"
var2 = "FNR"


res={}
x1_set=set()
x2_set=set()
for x in df.index:
    x1 = str(df.loc[x,var1])
    x2 = str(df.loc[x,var2])
    x1_set.add(x1)
    x2_set.add(x2)
    if not x1 in res:
        res[x1] = {}
    if not x2 in res[x1]:
        res[x1][x2] = []
    res[x1][x2].append(float(df.loc[x,"accuracy"]))

index1 = sorted(list(x1_set))
index2 = sorted(list(x2_set))

df=pd.DataFrame(index=index1,columns=index2,dtype=float)

for x1 in res:
    for x2 in res[x1]:
        mean_accuracy = np.mean(res[x1][x2])
        df.loc[x1,x2] = mean_accuracy


# get mean accuracy with MLP
sum_accuracies=0
count=0
for f in os.listdir(results_dirMLP):
    with open(os.path.join(results_dirMLP,f)) as json_file:
        data = json.load(json_file)
        sum_accuracies+=float(data["accuracy"])
        count+=1

MLP_accuracy = sum_accuracies/count

min_acc = np.min(np.min(df))
max_acc = np.max(np.max(df))

plt.rcParams.update({'font.size': 22})
ax=sns.heatmap(df,center = MLP_accuracy,cmap='RdYlBu',annot=True,cbar_kws= {"label":"Accuracy"},annot_kws={"size": 12}) #center = MLP_accuracy,cmap='RdYlBu_r'
cbar = ax.collections[0].colorbar
cbar.set_ticks([min_acc, MLP_accuracy, 0.5, max_acc])
cbar.set_ticklabels(["{:.2f}".format(min_acc), "MLP acc", "0.5", "{:.2f}".format(max_acc)])
#ax=sns.heatmap(df- MLP_accuracy,center=0,cmap='RdYlBu',annot=True,cbar_kws= {"label":"Accuracy"}) 
ax.set_ylabel("False Positive Rate (edges added)",x=-0.5)
ax.set_xlabel("False Negative Rate (true edges removed)")
ax.tick_params(labelsize=20)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.title("Accuracy with a GNN, when using a noisy graph with missing and false edges.",y=-0.1)
plt.show()


