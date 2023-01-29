import hydra
import random
import torch
import wandb
import numpy as np

from itertools import count
from typing import Any, Dict, Optional

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

def run_a2c(env, 
            obs, 
            agent_1, 
            agent_2, 
            target_steps, 
            reward_window, 
            device, 
            use_history, 
            is_pc, 
            n_actions):
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

            action_1 = torch.tensor([np.random.choice(n_actions, p=dist_1.cpu().detach().numpy())],
                                    requires_grad=False,
                                    device=device)
            action_2 = torch.tensor([np.random.choice(n_actions, p=dist_2.cpu().detach().numpy())],
                                    requires_grad=False, 
                                    device=device)
            log_prob_1 = torch.log(dist_1[action_1])
            log_prob_2 = torch.log(dist_2[action_2])

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
                    actions = torch.cat((action_1, action_2), 0)
                    agent_1.update_history(actions, curr_state)
                    agent_2.update_history(actions, curr_state)
                    next_state = agent_1.aggregate_history()
                    next_state = torch.tensor(next_state, dtype=torch.float32 , device=device)
                else:
                    next_state = torch.tensor(curr_state, dtype=torch.float32 , device=device)
            else:
                #print("rewards: ", rewards)
                next_state = None

            # Store the transition in memory
            if is_pc:
                agent_1.buffer.push(state, action_1, torch.unsqueeze(dist_1, 0).detach(), next_state, reward_1)
                agent_2.buffer.push(state, action_2, torch.unsqueeze(dist_2, 0).detach(), next_state, reward_2)

                # Store opponent's current policy
                agent_1.o_actor.load_state_dict(agent_2.actor.state_dict())
                agent_2.o_actor.load_state_dict(agent_1.actor.state_dict())
            
            agent_1.transition = [state, log_prob_1, next_state, reward_1, done, action_1, torch.unsqueeze(dist_2, 0).detach()]
            agent_2.transition = [state, log_prob_2, next_state, reward_2, done, action_2, torch.unsqueeze(dist_1, 0).detach()]

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            policy_loss_1, value_loss_1 = agent_1.optimize_model()
            policy_loss_2, value_loss_2 = agent_2.optimize_model()

            # Logging info
            cum_steps+=1
            wandb_info['cum_steps'] = cum_steps
            wandb_info['agent_1_avg_reward'] = avg_1
            wandb_info['agent_2_avg_reward'] = avg_2
            wandb_info['agent_1_policy_loss'] = policy_loss_1
            wandb_info['agent_1_value_loss'] = value_loss_1
            wandb_info['agent_2_policy_loss'] = policy_loss_2
            wandb_info['agent_2_value_loss'] = value_loss_2
            wandb_info['total_avg_reward'] = avg_1 + avg_2

            wandb.log(wandb_info)
            
            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_steps == 0:
            agent_1.t_net.load_state_dict(agent_1.q_net.state_dict())
            agent_2.t_net.load_state_dict(agent_2.q_net.state_dict())

def run_nash_ac(env, 
                obs, 
                agent_1, 
                agent_2, 
                target_steps, 
                reward_window, 
                device, 
                use_history,
                n_actions):
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
        state_his = []
        a_his = []
        b_his = []


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

            action_1 = torch.tensor([np.random.choice(n_actions, p=dist_1.cpu().detach().numpy())],
                                    requires_grad=False,
                                    device=device)
            action_2 = torch.tensor([np.random.choice(n_actions, p=dist_2.cpu().detach().numpy())],
                                    requires_grad=False, 
                                    device=device)
            log_prob_1 = torch.log(dist_1[action_1])
            log_prob_2 = torch.log(dist_2[action_2])

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
                    actions = torch.cat((action_1, action_2), 0)
                    agent_1.update_history(actions, curr_state)
                    agent_2.update_history(actions, curr_state)
                    next_state = agent_1.aggregate_history()
                    next_state = torch.tensor(next_state, dtype=torch.float32 , device=device)
                else:
                    next_state = torch.tensor(curr_state, dtype=torch.float32 , device=device)
            else:
                #print("rewards: ", rewards)
                next_state = None

            # Store the transition in memory
            agent_1.buffer.push(state, action_1, action_2, torch.unsqueeze(dist_1, 0).detach(), next_state, reward_1)
            agent_2.buffer.push(state, action_2, action_1, torch.unsqueeze(dist_2, 0).detach(), next_state, reward_2)
            agent_1.buffer.push(state, action_2, action_1, torch.unsqueeze(dist_2, 0).detach(), next_state, reward_2)
            agent_2.buffer.push(state, action_1, action_2, torch.unsqueeze(dist_1, 0).detach(), next_state, reward_1)

            # Store opponent's current policy
            agent_1.o_actor.load_state_dict(agent_2.actor.state_dict())
            agent_2.o_actor.load_state_dict(agent_1.actor.state_dict())
            
            agent_1.transition = [state, torch.unsqueeze(dist_2, 0).detach()]
            agent_2.transition = [state, torch.unsqueeze(dist_1, 0).detach()]

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            game_value_1, value_loss_1 = agent_1.optimize_model()
            game_value_2, value_loss_2 = agent_2.optimize_model()

            # Logging info
            cum_steps+=1
            wandb_info['cum_steps'] = cum_steps
            wandb_info['agent_1_avg_reward'] = avg_1
            wandb_info['agent_2_avg_reward'] = avg_2
            wandb_info['agent_1_game_value'] = game_value_1
            wandb_info['agent_1_value_loss'] = value_loss_1
            wandb_info['agent_2_game_value'] = game_value_2
            wandb_info['agent_2_value_loss'] = value_loss_2
            wandb_info['total_avg_reward'] = avg_1 + avg_2

            wandb.log(wandb_info)
            
            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_steps == 0:
            agent_1.t_net.load_state_dict(agent_1.q_net.state_dict())
            agent_2.t_net.load_state_dict(agent_2.q_net.state_dict())

def run_rf_nash_ac(env, 
                   obs, 
                   agent_1, 
                   agent_2, 
                   target_steps, 
                   reward_window, 
                   device, 
                   use_history,
                   n_actions):
    num_episodes = 1000000
    avg_reward_1 = []
    avg_reward_2 = []
    wandb_info = {}
    cum_steps = 0
    policy_hist_len = agent_1.policy_hist_len

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        episode_durations = []
        episode_rewards = []
        obs, _ = env.reset()
        state =  torch.tensor(obs, dtype=torch.float32 , device=device)
        action_hist_a = []
        action_hist_b = []
        state_hist = []
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

            action_1 = torch.tensor([np.random.choice(n_actions, p=dist_1.cpu().detach().numpy())],
                                    requires_grad=False,
                                    device=device)
            action_2 = torch.tensor([np.random.choice(n_actions, p=dist_2.cpu().detach().numpy())],
                                    requires_grad=False, 
                                    device=device)
            log_prob_1 = torch.log(dist_1[action_1])
            log_prob_2 = torch.log(dist_2[action_2])

            # Observe new state
            curr_state, rewards, done, _, _ = env.step((action_1.item(), action_2.item()))
            reward_1 = torch.tensor([rewards[0]], dtype=torch.float32 , device=device)
            reward_2 = torch.tensor([rewards[1]], dtype=torch.float32 , device=device)

            avg_reward_1.insert(0, rewards[0])
            avg_reward_2.insert(0, rewards[1])

            avg_reward_1 = avg_reward_1[0:reward_window]
            avg_reward_2 = avg_reward_2[0:reward_window]

            avg_1 = sum(avg_reward_1)/len(avg_reward_1)
            avg_2 = sum(avg_reward_2)/len(avg_reward_2)

            if not done:
                # Aggregate histories if history is used
                if use_history:
                    actions = torch.cat((action_1, action_2), 0)
                    agent_1.update_history(actions, curr_state)
                    agent_2.update_history(actions, curr_state)
                    next_state = agent_1.aggregate_history()
                    next_state = torch.tensor(next_state, dtype=torch.float32 , device=device)
                else:
                    next_state = torch.tensor(curr_state, dtype=torch.float32 , device=device)
            else:
                #print("rewards: ", rewards)
                next_state = None

            # Store the transition in memory
            agent_1.buffer.push(state, action_1, action_2, torch.unsqueeze(dist_1, 0).detach(), next_state, reward_1)
            agent_2.buffer.push(state, action_2, action_1, torch.unsqueeze(dist_2, 0).detach(), next_state, reward_2)

            # Update histories
            action_hist_a.insert(0, action_1)
            action_hist_b.insert(0, action_2)
            state_hist.insert(0, state)

            action_hist_a = action_hist_a[0:policy_hist_len]
            action_hist_b = action_hist_b[0:policy_hist_len]
            state_hist = state_hist[0:policy_hist_len]

            # Store the policy transition in memory
            if len(action_hist_a) == policy_hist_len:
                agent_1.policy_buffer.push(torch.cat(state_hist).to(device), 
                                           torch.cat(action_hist_a).to(device), 
                                           torch.cat(action_hist_b).to(device))
                agent_2.policy_buffer.push(torch.cat(state_hist).to(device), 
                                           torch.cat(action_hist_b).to(device), 
                                           torch.cat(action_hist_a).to(device))

            # Store opponent's current policy
            agent_1.o_actor.load_state_dict(agent_2.actor.state_dict())
            agent_2.o_actor.load_state_dict(agent_1.actor.state_dict())
            
            agent_1.transition = [state, torch.unsqueeze(dist_2, 0).detach()]
            agent_2.transition = [state, torch.unsqueeze(dist_1, 0).detach()]

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the reward and policy networks)
            game_value_1, value_loss_1 = agent_1.optimize_model()
            game_value_2, value_loss_2 = agent_2.optimize_model()

            # Logging info
            cum_steps+=1
            wandb_info['cum_steps'] = cum_steps
            wandb_info['agent_1_avg_reward'] = avg_1
            wandb_info['agent_2_avg_reward'] = avg_2
            wandb_info['agent_1_game_value'] = game_value_1
            wandb_info['agent_1_value_loss'] = value_loss_1
            wandb_info['agent_2_game_value'] = game_value_2
            wandb_info['agent_2_value_loss'] = value_loss_2
            wandb_info['total_avg_reward'] = avg_1 + avg_2

            wandb.log(wandb_info)
            
            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_steps == 0:
            agent_1.t_net.load_state_dict(agent_1.q_net.state_dict())
            agent_2.t_net.load_state_dict(agent_2.q_net.state_dict())
