#!/bin/sh



for seed in 0 1 2 3 4 5 6 7 8 9; do


for FPR in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
for FNR in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
for infer_graph in False; do
for classifier in GraphSAGE; do
for n_hidden_GNN in 8; do
	bsub -W 1:00 -R "rusage[mem=8000]" "python run.py -o out_graphQual --classifier $classifier --n_hidden_GNN $n_hidden_GNN --infer_graph $infer_graph --seed $seed --FPR $FPR --FNR $FNR"
done
done
done
done
done

for classifier in MLP; do
for n_hidden_FC in 30; do
	bsub -W 1:00 -R "rusage[mem=4000]" "python run.py -o out_graphQualMLP --classifier $classifier --n_hidden_FC $n_hidden_FC --infer_graph False --seed $seeds"
done
done

done

