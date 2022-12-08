#!/bin/bash
SEEDS=(17)
EPS_DECAYS=(500 1000 5000 10000)
HIST_LENS=(2 3 4 5)

for seed in ${SEEDS[@]}; do
    for eps_decay in ${EPS_DECAYS[@]}; do
        for hist_len in ${HIST_LENS[@]}; do
            export WANDB_NAME=dqn_hist_seed_${seed}_eps_decay_${eps_decay}_len_${hist_len}
            sbatch --job-name=cogames_dqn_${game} run_dqn_hist.slurm ${seed} ${eps_decay} ${hist_len}
        done
    done
done