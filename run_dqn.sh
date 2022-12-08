#!/bin/bash
SEEDS=(1 2)
EPS_DECAYS=(5000 10000 50000)

for seed in ${SEEDS[@]}; do
    for eps_decay in ${EPS_DECAYS[@]}; do
        export WANDB_NAME=dqn_seed_${seed}_eps_decay_${eps_decay}
        sbatch --job-name=cogames_dqn_${game} run_dqn.slurm ${seed} ${eps_decay}
    done
done