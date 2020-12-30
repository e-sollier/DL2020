#!/bin/sh

for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    for tag in pbmc_2020_12_04 cross_hm_2; do
        for n_obs_train in 125 250 500 1000; do
            for classifier in MLP; do
                for n_hidden_FC in 0 8 32 64;do
                    bsub -W 1:00 -R "rusage[mem=4000]" \
                    "python run.py -i ../data_input -tag $tag \
                    --classifier $classifier \
                    --n_hidden_FC $n_hidden_FC \
                    --dropout 0.1 \
                    --n_obs_train $n_obs_train \
                    --infer_graph True \
                    --seed $seed"
                done
            done
            for alpha in 0.5 1 1.5 2 3 4 8; do
                for classifier in Chebnet GraphSAGE; do
                    for K in 1 2 4 8; do
                        for n_hidden_GNN in 4 8 16; do
                            bsub -W 1:00 -R "rusage[mem=6000]" \
                            "python run.py -i ../data_input -tag $tag\ 
                            --classifier $classifier \
                            --n_hidden_GNN $n_hidden_GNN \
                            --dropout 0.1 \
                            --alpha $alpha \
                            --n_obs_train $n_obs_train \
                            --infer_graph True \
                            --seed $seed
                            --K $K"
                        done
                    done
                done
            done
        done
    done
done



