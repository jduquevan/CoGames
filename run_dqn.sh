#!/bin/bash
echo "${BASH_VERSION}"
SEEDS=(0 1 2)

for seed in ${SEEDS[@]}; do
    export WANDB_NAME=dqn_${seed}
    sbatch --job-name=cogames_dqn_${game} run_dqn.slurm ${seed}
done