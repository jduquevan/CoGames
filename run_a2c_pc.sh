#!/bin/bash
SEEDS=(13 21 23 29)
HIST_LENS=(2)
TEMPS=(0.25 0.5 1 2 4)

for seed in ${SEEDS[@]}; do
    for hist_len in ${HIST_LENS[@]}; do
        for temp in ${TEMPS[@]}; do
            export WANDB_NAME=a2c_pc_hist_seed_${seed}_len_${hist_len}_temp_${temp}
            sbatch --job-name=cogames_a2c run_a2c_pc.slurm ${seed} ${hist_len} ${temp}
        done
    done
done