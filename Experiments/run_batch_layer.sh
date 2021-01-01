#!/bin/sh

for seed in 0 1 2 3 4 5 6 7 8 9 ; do
for n_obs_train in 100 500 1000 3000; do


for infer_graph in False; do
for classifier in GraphSAGE Chebnet GINConv GATConv TransformerConv GraphConv MFConv; do
for n_hidden_GNN in 8; do
	bsub -W 1:00 -R "rusage[mem=6000]" "python run.py --oo out_layer --classifier $classifier --n_hidden_GNN $n_hidden_GNN --n_obs_train $n_obs_train --infer_graph $infer_graph --seed $seed"
done
done
done

done
done
