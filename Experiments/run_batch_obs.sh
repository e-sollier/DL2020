#!/bin/sh

for seed in 0 1 2 3 4 5 6 7 8 9 ; do
for n_obs_train in 50 100 500 1000 2500 5000 7500 10000 20000; do


for classifier in MLP; do
for n_hidden_FC in 0 40;do
	bsub -W 1:00 -R "rusage[mem=6000]" "python run.py -o out_obs --classifier $classifier --n_hidden_FC $n_hidden_FC --n_obs_train $n_obs_train --seed $seed"
done
done

bsub -W 1:00 -R "rusage[mem=6000]" "python run.py -o out_obs --classifier MLP --n_hidden_FC 80 --n_hidden_FC2 40 --n_obs_train $n_obs_train --seed $seed"

bsub -W 1:00 -R "rusage[mem=10000]" "python run.py -o out_obs --classifier MLP --n_hidden_FC 100 --n_hidden_FC2 80 --n_hidden_FC3 60 --n_obs_train $n_obs_train --seed $seed "

for infer_graph in True False; do
for classifier in GraphSAGE; do
for n_hidden_GNN in 8; do
	bsub -W 1:00 -R "rusage[mem=10000]" "python run.py -o out_obs --classifier $classifier --n_hidden_GNN $n_hidden_GNN --n_obs_train $n_obs_train --infer_graph $infer_graph --seed $seed"
done
done
done

done
done
