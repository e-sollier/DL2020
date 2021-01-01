#!/bin/sh

for seed in 0 1 2 3 4 5 6 7 8 9 ; do
for n_obs_train in 2000; do
for n_features in 75 100 150 200 300 500; do



bsub -W 1:00 -R "rusage[mem=4000]" "python run.py --oo out_features --classifier MLP --n_hidden_FC 40 --n_obs_train $n_obs_train --seed $seed --n_features $n_features"


bsub -W 1:00 -R "rusage[mem=6000]" "python run.py --oo out_features --classifier GraphSAGE --n_hidden_GNN 8 --n_obs_train $n_obs_train --infer_graph False --seed $seed --n_features $n_features"

bsub -W 4:00 -R "rusage[mem=10000]" "python run.py --oo out_features --classifier GraphSAGE --n_hidden_GNN 8 --n_obs_train $n_obs_train --infer_graph True --CV_alpha True --seed $seed --n_features $n_features"


done
done
done
