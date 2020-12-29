#!/bin/sh


for i in activation diffusion; do
    for graph_model in BA ER; do
        for signal_test in 10 5 3; do
            for n_features in 250 125 50; do
                for n_char_features in 25 10 5; do
                    # for classifier in MLP; do
                    #     for n_hidden_FC in 8 16 32;do
                    #         bsub -W 3:00 -R "rusage[mem=4000, ngpus_excl_p=1]" \
                    #         "python run.py -i $i -o out \
                    #         --classifier $classifier \
                    #         --n_hidden_FC $n_hidden_FC \
                    #         --dropout 0.1 \
                    #         --signal_test $signal_test \
                    #         --n_features $n_features \
                    #         --n_char_features $n_char_features\
                    #         --graph_model $graph_model"
                    #     done
                    # done
                    for classifier in Chebnet; do
                        for K in 1 2 4 8; do
                            for n_hidden_GNN in 4 8 16; do
                                bsub -W 3:00 -R "rusage[mem=4000, ngpus_excl_p=1]" \
                                "python run.py -i $i -o out \
                                --classifier $classifier \
                                --n_hidden_GNN $n_hidden_GNN \
                                --K $K \
                                --dropout 0.1 \
                                --signal_test $signal_test \
                                --n_features $n_features \
                                --n_char_features $n_char_features\
                                --graph_model $graph_model"
                            done
                        done
                    done
                done
            done
        done
    done
done

