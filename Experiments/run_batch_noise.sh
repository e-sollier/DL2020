#!/bin/sh

for seed in 0 1 2 3 4 5 6 7 8 9 ; do
for n_obs_train in 1000; do
for noise in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do


for classifier in MLP; do
for n_hidden_FC in 40;do
	bsub -W 1:00 -R "rusage[mem=4000]" "python run.py --oo out_noise --classifier $classifier --n_hidden_FC $n_hidden_FC --n_obs_train $n_obs_train --seed $seed --noise_train $noise --noise_test  $noise"
done
done


for infer_graph in True False; do
for classifier in GraphSAGE; do
for n_hidden_GNN in 8; do
	bsub -W 1:00 -R "rusage[mem=6000]" "python run.py --oo out_noise --classifier $classifier --n_hidden_GNN $n_hidden_GNN --n_obs_train $n_obs_train --infer_graph $infer_graph --seed $seed --noise_train $noise --noise_test  $noise"
done
done
done

done
done
done
