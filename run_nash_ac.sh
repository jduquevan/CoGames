#!/bin/bash
SEEDS=(1 3 5 7)
HIST_LENS=(3)
TEMPS=(6 8)
LRS=(0.0001 0.0005 0.001 0.005)

for seed in ${SEEDS[@]}; do
    for hist_len in ${HIST_LENS[@]}; do
        for temp in ${TEMPS[@]}; do
            for lr in ${LRS[@]}; do
                export WANDB_NAME=nash_ac_hist_seed_${seed}_len_${hist_len}_temp_${temp}_lr_${lr}
                sbatch --job-name=cogames_nash_ac run_nash_ac.slurm ${seed} ${hist_len} ${temp} ${lr}
            done
        done
    done
done