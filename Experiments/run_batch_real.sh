#!/bin/sh

for seed in 0 1 2 3 4 5 6 7 8 9; do
    for tag in pbmc cross; do
        for n_obs_train in 125 250 500 1000; do
            for classifier in MLP; do
                for n_hidden_FC in 0 8 32 64;do
                    bsub -W 1:00 -R "rusage[mem=6000]" -o MLP1\
                    "python run.py -i ../data_input -tag $tag \
                    --classifier $classifier \
                    --n_hidden_FC $n_hidden_FC \
                    --dropout 0.1 \
                    --n_obs_train $n_obs_train \
                    --alpha 5 \
                    --infer_graph True \
                    --seed $seed" 
                    for n_hidden_FC2 in 4 8 16;do
                        bsub -W 1:00 -R "rusage[mem=6000]" -o MLP2\
                        "python run.py -i ../data_input -tag $tag \
                        --classifier $classifier \
                        --n_hidden_FC $n_hidden_FC \
                        --n_hidden_FC2 $n_hidden_FC2 \
                        --dropout 0.1 \
                        --alpha 5 \
                        --n_obs_train $n_obs_train \
                        --infer_graph True \
                        --seed $seed" 
                        for n_hidden_FC3 in 4 8 16;do
                            bsub -W 1:00 -R "rusage[mem=7000]" -o MLP3\
                            "python run.py -i ../data_input -tag $tag \
                            --classifier $classifier \
                            --n_hidden_FC $n_hidden_FC \
                            --n_hidden_FC2 $n_hidden_FC2 \
                            --n_hidden_FC3 $n_hidden_FC3 \
                            --dropout 0.1 \
                            --alpha 5 \
                            --n_obs_train $n_obs_train \
                            --infer_graph True \
                            --seed $seed" 
                        done
                    done 
                done
            done
            for alpha in 0.5 1 1.5 2 3 4 8; do
                for n_hidden_GNN in 4 8 16 32; do
                    for classifier in Chebnet; do
                        for K in 1 2 4 8; do
                            bsub -W 1:00 -R "rusage[mem=6000]" -o Chebnet \
                            "python run.py -i ../data_input -tag $tag \
                            --classifier $classifier \
                            --n_hidden_GNN $n_hidden_GNN \
                            --dropout 0.1 \
                            --alpha $alpha \
                            --n_obs_train $n_obs_train \
                            --infer_graph True \
                            --seed $seed \
                            --K $K" 
                        done
                    done
                    # for classifier in GraphSAGE; do
                    #     bsub -W 1:00 -R "rusage[mem=6000]" -o GraphSAGE \
                    #     "python run.py -i ../data_input -tag $tag \ 
                    #     --classifier $classifier \
                    #     --n_hidden_GNN $n_hidden_GNN \
                    #     --dropout 0.1 \
                    #     --alpha $alpha \
                    #     --n_obs_train $n_obs_train \
                    #     --infer_graph True \
                    #     --seed $seed" 
                    # done
                done
            done
        done    
    done
done
