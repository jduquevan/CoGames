#!/bin/bash
echo "${BASH_VERSION}"
SEEDS=(1)
EPS_DECAYS=(500 1000 5000 10000 50000)

for seed in ${SEEDS[@]}; do
    for eps_decay in ${EPS_DECAYS[@]}; do
        export WANDB_NAME=dqn_seed_${seed}_eps_decay_${eps_decay}
        sbatch --job-name=cogames_dqn_${game} run_dqn.slurm ${seed} ${eps_decay}
    done
done