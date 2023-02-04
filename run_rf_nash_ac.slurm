#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=co_games_nash

# Remove one # to uncommment
#SBATCH --output=/network/scratch/j/juan.duque/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/j/juan.duque/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=0-02:59:00   
#SBATCH --gres=gpu:1 

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=NONE
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=juan.duque@mila.quebec


# Submit jobs.
version=4
export WANDB_ENTITY="jduque"

module purge
eval "$(conda shell.bash hook)"
conda activate cogames

python -m src.run seed=${1} agent_type="s_rf_nash_ac" use_history=True  base_agent.use_actions=True \
    base_agent.model_type="lstm" base_agent.history_len=${2} rf_nash_ac_agent.temperature=${3} optim.lr=${4} \
    rf_nash_ac_agent.policy_hist_len=${5} rf_nash_ac_agent.is_p_pc=True

# python -m src.run seed=1 agent_type="s_rf_nash_ac" use_history=True base_agent.use_actions=True base_agent.model_type="lstm" base_agent.history_len=2 rf_nash_ac_agent.temperature=2 rf_nash_ac_agent.is_p_pc=True
# python -m src.run seed=1 agent_type="s_rf_nash_ac" use_history=True base_agent.use_actions=True base_agent.model_type="lstm" rf_nash_ac_agent.temperature=8 optim.lr=0.0001 base_agent.history_len=2 rf_nash_ac_agent.temperature=2 rf_nash_ac_agent.is_p_pc=True