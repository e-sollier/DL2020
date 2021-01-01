#!/bin/sh


for n_obs_train in 100 200 500 1000 5000; do
for infer_graph in True; do
for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
for alpha in 0.1 0.5 1 1.5 2 2.5 3 3.5 4 5 6 8 10 15; do

for classifier in GraphSAGE; do
for n_hidden_GNN in 8; do
	bsub -W 1:00 -R "rusage[mem=8000]" "python run.py --oo out_alpha --classifier $classifier --n_hidden_GNN $n_hidden_GNN --n_obs_train $n_obs_train --infer_graph $infer_graph --seed $seed --alpha $alpha"
done
done

done
done
done
done
