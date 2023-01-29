import hydra
import os
import random
import sys
import torch
import wandb
import numpy as np

from itertools import count
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional

from .coin_game import CoinGame
from .agents import DQNAgent, A2CAgent, A2CPCAgent, NashActorCriticAgent, ReinforcedNashActorCriticAgent
from .algos import run_dqn, run_a2c, run_nash_ac, run_rf_nash_ac

N_ACTIONS = 4

@hydra.main(config_path="../scripts", config_name="config")
def main(args: DictConfig):

    config: Dict[str, Any] = OmegaConf.to_container(args, resolve=True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    wandb.init(config=config, dir="/network/scratch/j/juan.duque/wandb/", project="Co-games", reinit=True, anonymous="allow")

    env = CoinGame(2, 1)
    obs, _ = env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_steps = config["target_steps"]
    reward_window = config["reward_window"]
    use_history = config["use_history"]
    is_pc = config["is_pc"]

    if config["agent_type"] == "dqn":
        agent_1 = DQNAgent(config["base_agent"],
                           config["dqn_agent"], 
                           device=device,
                           n_actions=N_ACTIONS,
                           obs_shape=obs.shape)

        agent_2 = DQNAgent(config["base_agent"],
                           config["dqn_agent"], 
                           device=device,
                           n_actions=N_ACTIONS,
                           obs_shape=obs.shape)
        
        run_dqn(env=env, 
                obs=obs, 
                agent_1=agent_1, 
                agent_2=agent_2, 
                target_steps=target_steps, 
                reward_window=reward_window, 
                device=device, 
                use_history=use_history)

    elif config["agent_type"] == "a2c":
        # TODO: Debug A2C agents and initialization 
        if is_pc:
            agent_1 = A2CPCAgent(config["base_agent"],
                                 config["a2c_agent"], 
                                 device=device,
                                 n_actions=N_ACTIONS,
                                 obs_shape=obs.shape)

            agent_2 = A2CPCAgent(config["base_agent"],
                                 config["a2c_agent"], 
                                 device=device,
                                 n_actions=N_ACTIONS,
                                 obs_shape=obs.shape)
        else:
            agent_1 = A2CAgent(config["base_agent"],
                               config["a2c_agent"], 
                               device=device,
                               n_actions=N_ACTIONS,
                               obs_shape=obs.shape)

            agent_2 = A2CAgent(config["base_agent"],
                               config["a2c_agent"], 
                               device=device,
                               n_actions=N_ACTIONS,
                               obs_shape=obs.shape)

        run_a2c(env=env, 
                obs=obs, 
                agent_1=agent_1, 
                agent_2=agent_2, 
                target_steps=target_steps, 
                reward_window=reward_window, 
                device=device, 
                use_history=use_history,
                is_pc=is_pc,
                n_actions=N_ACTIONS)

    elif config["agent_type"] == "nash_ac":
        agent_1 = NashActorCriticAgent(config["nash_ac_agent"],
                                       config["sgd"], 
                                       device=device,
                                       n_actions=N_ACTIONS,
                                       obs_shape=obs.shape)

        agent_2 = NashActorCriticAgent(config["nash_ac_agent"],
                                       config["sgd"], 
                                       device=device,
                                       n_actions=N_ACTIONS,
                                       obs_shape=obs.shape)

        run_nash_ac(env=env, 
                    obs=obs, 
                    agent_1=agent_1, 
                    agent_2=agent_2, 
                    target_steps=target_steps, 
                    reward_window=reward_window, 
                    device=device, 
                    use_history=use_history,
                    n_actions=N_ACTIONS)

    elif config["agent_type"] == "rf_nash_ac":
        agent_1 = ReinforcedNashActorCriticAgent(config["base_agent"],
                                                 config["sgd"], 
                                                 device=device,
                                                 n_actions=N_ACTIONS,
                                                 obs_shape=obs.shape,
                                                 policy_hist_len=config["rf_nash_ac_agent"]["policy_hist_len"])

        agent_2 = ReinforcedNashActorCriticAgent(config["base_agent"],
                                                 config["sgd"], 
                                                 device=device,
                                                 n_actions=N_ACTIONS,
                                                 obs_shape=obs.shape,
                                                 policy_hist_len=config["rf_nash_ac_agent"]["policy_hist_len"])

        run_rf_nash_ac(env=env, 
                       obs=obs, 
                       agent_1=agent_1, 
                       agent_2=agent_2, 
                       target_steps=target_steps, 
                       reward_window=reward_window, 
                       device=device, 
                       use_history=use_history,
                       n_actions=N_ACTIONS)
        

if __name__ == "__main__":
    main()