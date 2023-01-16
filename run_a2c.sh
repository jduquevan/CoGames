#!/bin/bash
SEEDS=(17 21 23 25 29)
HIST_LENS=(2 3 4 5)

for seed in ${SEEDS[@]}; do
    for hist_len in ${HIST_LENS[@]}; do
        export WANDB_NAME=a2c_hist_seed_${seed}_len_${hist_len}
        sbatch --job-name=cogames_a2c run_a2c.slurm ${seed} ${hist_len}
    done
done