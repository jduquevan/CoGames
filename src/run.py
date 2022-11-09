import hydra
import os
import sys
import torch
import numpy as np

from itertools import count
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional

from .coin_game import CoinGame
from .agents import DQNAgent


@hydra.main(config_path="../scripts", config_name="config")
def main(args: DictConfig):

    config: Dict[str, Any] = OmegaConf.to_container(args, resolve=True)
    print(config["seed"])
    print(type(config["seed"]))
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    env = CoinGame(2, 1)
    obs, _ = env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_1 = DQNAgent(**config["dqn_agent"], 
                     device=device,
                     n_actions=4,
                     obs_shape=obs.shape)

    agent_2 = DQNAgent(**config["dqn_agent"], 
                     device=device,
                     n_actions=4,
                     obs_shape=obs.shape)

    target_steps = config["target_steps"]
    
    num_episodes = 5000
    avg_reward_1 = []
    avg_reward_2 = []
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        episode_durations = []
        episode_rewards = []
        obs, _ = env.reset()
        state =  torch.tensor(obs, dtype=torch.float32 , device=device)
        for t in count():
            # Select and perform an action
            action_1 = agent_1.select_action(state)
            action_2 = agent_2.select_action(state)

            curr_state, rewards, done, _, _ = env.step((action_1.item(), action_2.item()))
            reward_1 = torch.tensor([rewards[0]], dtype=torch.float32 , device=device)
            reward_2 = torch.tensor([rewards[1]], dtype=torch.float32 , device=device)

            avg_reward_1.insert(0, rewards[0])
            avg_reward_2.insert(0, rewards[1])

            avg_reward_1 = avg_reward_1[0:10]
            avg_reward_2 = avg_reward_2[0:10]

            avg_1 = sum(avg_reward_1)/len(avg_reward_1)
            avg_2 = sum(avg_reward_2)/len(avg_reward_2)

            # Observe new state
            if not done:
                next_state = torch.tensor(curr_state, dtype=torch.float32 , device=device)
            else:
                #print("rewards: ", rewards)
                next_state = None

            # Store the transition in memory
            agent_1.buffer.push(state, action_1, next_state, reward_1)
            agent_2.buffer.push(state, action_2, next_state, reward_2)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent_1.optimize_model()
            agent_2.optimize_model()
            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_steps == 0:
            agent_1.t_net.load_state_dict(agent_1.q_net.state_dict())
            agent_2.t_net.load_state_dict(agent_2.q_net.state_dict())



if __name__ == "__main__":
    main()