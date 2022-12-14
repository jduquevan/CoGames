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
from .agents import DQNAgent, A2CAgent

N_ACTIONS = 4

def run_dqn(env, obs, agent_1, agent_2, target_steps, reward_window, device, use_history):
    num_episodes = 100000
    avg_reward_1 = []
    avg_reward_2 = []
    wandb_info = {}
    cum_steps = 0

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        episode_durations = []
        episode_rewards = []
        obs, _ = env.reset()
        state =  torch.tensor(obs, dtype=torch.float32 , device=device)

        actions = torch.tensor([-1, -1])
        agent_1.update_history(actions, state)
        agent_2.update_history(actions, state)

        for t in count():
            if use_history:
                state = agent_1.aggregate_history().float()
                state = torch.unsqueeze(state, dim=0)
            # Select and perform an action
            action_1 = agent_1.select_action(state)
            action_2 = agent_2.select_action(state)

            curr_state, rewards, done, _, _ = env.step((action_1.item(), action_2.item()))
            reward_1 = torch.tensor([rewards[0]], dtype=torch.float32 , device=device)
            reward_2 = torch.tensor([rewards[1]], dtype=torch.float32 , device=device)

            avg_reward_1.insert(0, rewards[0])
            avg_reward_2.insert(0, rewards[1])

            avg_reward_1 = avg_reward_1[0:reward_window]
            avg_reward_2 = avg_reward_2[0:reward_window]

            avg_1 = sum(avg_reward_1)/len(avg_reward_1)
            avg_2 = sum(avg_reward_2)/len(avg_reward_2)

            # Observe new state
            if not done:
                # Aggregate histories if history is used
                if use_history:
                    actions = torch.tensor([action_1, action_2])
                    agent_1.update_history(actions, curr_state)
                    agent_2.update_history(actions, curr_state)
                    next_state = agent_1.aggregate_history()
                else:
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
            loss_1 = agent_1.optimize_model()
            loss_2 = agent_2.optimize_model()

            # Logging info
            cum_steps+=1
            wandb_info['cum_steps'] = cum_steps
            wandb_info['agent_1_avg_reward'] = avg_1
            wandb_info['agent_2_avg_reward'] = avg_2
            wandb_info['agent_1_loss'] = loss_1
            wandb_info['agent_2_loss'] = loss_2
            wandb_info['total_avg_reward'] = avg_1 + avg_2

            wandb.log(wandb_info)
            
            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_steps == 0:
            agent_1.t_net.load_state_dict(agent_1.q_net.state_dict())
            agent_2.t_net.load_state_dict(agent_2.q_net.state_dict())

def run_a2c(env, obs, agent_1, agent_2, target_steps, reward_window, device, use_history):
    num_episodes = 100000
    avg_reward_1 = []
    avg_reward_2 = []
    wandb_info = {}
    cum_steps = 0

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        episode_durations = []
        episode_rewards = []
        obs, _ = env.reset()
        state =  torch.tensor(obs, dtype=torch.float32 , device=device)

        actions = torch.tensor([-1, -1])
        agent_1.update_history(actions, state)
        agent_2.update_history(actions, state)

        for t in count():
            if use_history:
                state = agent_1.aggregate_history().float()
                state = torch.unsqueeze(state, dim=0)
            # Select and perform an action
            dist_1 = agent_1.select_action(state)
            dist_2 = agent_2.select_action(state)

            action_1 = torch.tensor([np.random.choice(N_ACTIONS, p=dist_1.cpu().detach().numpy())], device=device)
            action_2 = torch.tensor([np.random.choice(N_ACTIONS, p=dist_2.cpu().detach().numpy())], device=device)
            log_prob_1 = torch.tensor(torch.log(dist_1[action_1]), dtype=torch.float32 , device=device)
            log_prob_2 = torch.tensor(torch.log(dist_2[action_2]), dtype=torch.float32 , device=device)

            curr_state, rewards, done, _, _ = env.step((action_1.item(), action_2.item()))
            reward_1 = torch.tensor([rewards[0]], dtype=torch.float32 , device=device)
            reward_2 = torch.tensor([rewards[1]], dtype=torch.float32 , device=device)

            avg_reward_1.insert(0, rewards[0])
            avg_reward_2.insert(0, rewards[1])

            avg_reward_1 = avg_reward_1[0:reward_window]
            avg_reward_2 = avg_reward_2[0:reward_window]

            avg_1 = sum(avg_reward_1)/len(avg_reward_1)
            avg_2 = sum(avg_reward_2)/len(avg_reward_2)

            # Observe new state
            if not done:
                # Aggregate histories if history is used
                if use_history:
                    actions = torch.tensor([action_1, action_2])
                    agent_1.update_history(actions, curr_state)
                    agent_2.update_history(actions, curr_state)
                    next_state = agent_1.aggregate_history()
                else:
                    next_state = torch.tensor(curr_state, dtype=torch.float32 , device=device)
            else:
                #print("rewards: ", rewards)
                next_state = None

            # Store the transition in memory
            agent_1.buffer.push(state, action_1, log_prob_1, dist_1, next_state, reward_1)
            agent_2.buffer.push(state, action_2, log_prob_2, dist_2, next_state, reward_2)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss_1 = agent_1.optimize_model()
            loss_2 = agent_2.optimize_model()

            # Logging info
            cum_steps+=1
            wandb_info['cum_steps'] = cum_steps
            wandb_info['agent_1_avg_reward'] = avg_1
            wandb_info['agent_2_avg_reward'] = avg_2
            wandb_info['agent_1_loss'] = loss_1
            wandb_info['agent_2_loss'] = loss_2
            wandb_info['total_avg_reward'] = avg_1 + avg_2

            wandb.log(wandb_info)
            
            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_steps == 0:
            agent_1.t_net.load_state_dict(agent_1.q_net.state_dict())
            agent_2.t_net.load_state_dict(agent_2.q_net.state_dict())

@hydra.main(config_path="../scripts", config_name="config")
def main(args: DictConfig):

    config: Dict[str, Any] = OmegaConf.to_container(args, resolve=True)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    wandb.init(config=config, project="Co-games", reinit=True, anonymous="allow")

    env = CoinGame(2, 1)
    obs, _ = env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_steps = config["target_steps"]
    reward_window = config["reward_window"]
    use_history = config["use_history"]

    if config["agent_type"] == "dqn":
        agent_1 = DQNAgent(config["dqn_agent"], 
                           device=device,
                           n_actions=N_ACTIONS,
                           obs_shape=obs.shape)

        agent_2 = DQNAgent(config["dqn_agent"], 
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
        agent_1 = A2CAgent(config["a2c_agent"], 
                           device=device,
                           n_actions=N_ACTIONS,
                           obs_shape=obs.shape)

        agent_2 = A2CAgent(config["a2c_agent"], 
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
                use_history=use_history)

if __name__ == "__main__":
    main()