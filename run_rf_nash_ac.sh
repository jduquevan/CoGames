#!/bin/bash
SEEDS=(5 3)
HIST_LENS=(2)
TEMPS=(6 8)
POLICY_LENS=(8 16 24)
LRS=(0.0001 0.0005 0.001 0.005)

for seed in ${SEEDS[@]}; do
    for hist_len in ${HIST_LENS[@]}; do
        for temp in ${TEMPS[@]}; do
            for lr in ${LRS[@]}; do
                for p_len in ${POLICY_LENS[@]}; do
                    export WANDB_NAME=rf_nash_ac_hist_seed_${seed}_len_${hist_len}_temp_${temp}_lr_${lr}
                    sbatch --job-name=cogames_rf_nash_ac run_rf_nash_ac.slurm ${seed} ${hist_len} ${temp} ${lr} ${p_len}
                done
            done
        done
    done
done