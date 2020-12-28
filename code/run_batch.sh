#!/bin/sh


for i in activation; do
for signal in 4; do
for n_obs_train in 500; do
for n_features in 50; do
for infer_graph in True; do
for seed in 0 1 2 3; do


for classifier in MLP; do
for n_hidden_FC in 0 30;do
	bsub -W 4:00 -R "rusage[mem=4000]" "python run.py -i $i -o out --classifier $classifier --n_hidden_FC $n_hidden_FC --signal_train $signal --signal_test $signal --n_obs_train $n_obs_train --infer_graph $infer_graph --dropout 0.2 --seed $seed"
done
done

for classifier in GraphSAGE Chebnet; do
for n_hidden_GNN in 4; do
	bsub -W 10:00 -R "rusage[mem=8000]" "python run.py -i $i -o out --classifier $classifier --n_hidden_GNN $n_hidden_GNN --signal_train $signal --signal_test $signal --n_obs_train $n_obs_train --infer_graph $infer_graph --dropout 0.2 --seed $seed"
done
done

done
done
done
done
done
done
