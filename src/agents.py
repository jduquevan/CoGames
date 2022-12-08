import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from functools import reduce

from .models import MLPModel, ConvModel, ReplayMemory, Transition, A2CTransition, Actor

class BaseAgent():
    def __init__(self,
                 device,
                 gamma,
                 n_actions,
                 eps_init,
                 eps_final,
                 eps_decay,
                 hidden_size,
                 num_layers,
                 batch_norm, 
                 obs_shape,
                 buffer_size, 
                 model_type="mlp",
                 opt_type="rmsprop",
                 batch_size=128,
                 history_len=0):
        self.steps_done = 0
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.eps_init = eps_init
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.obs_shape = obs_shape
        self.batch_size = batch_size
        self.history_len = history_len
        self.buffer_size =buffer_size
        self.model_type = model_type.lower()
        self.obs_size = history_len * reduce(lambda a, b: a * b, self.obs_shape)

        self.obs_history = []
        self.act_history = []
        
        if self.model_type == "mlp":
            self.q_net = MLPModel(in_size=self.obs_size,
                                  out_size=self.n_actions,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_norm=self.batch_norm)
            self.t_net = MLPModel(in_size=self.obs_size,
                                  out_size=self.n_actions,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_norm=self.batch_norm)
        elif self.model_type == "conv":
            self.q_net = ConvModel(in_channels=history_len,
                                  h=self.obs_shape[0],
                                  w=self.obs_shape[1],
                                  out_size=self.n_actions)
            self.t_net = ConvModel(in_channels=history_len,
                                  h=self.obs_shape[0],
                                  w=self.obs_shape[1],
                                  out_size=self.n_actions)
        else:
            raise Exception(self.model_type, " is not a supported model type for Agents")

        if opt_type.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.q_net.parameters())
        
        self.t_net.load_state_dict(self.q_net.state_dict())
        self.q_net.to(self.device)
        self.t_net.to(self.device)

    def update_history(self, act, state):
        self.act_history.insert(0, torch.tensor(act).to(self.device))
        self.obs_history.insert(0,  torch.tensor(state).to(self.device))
        self.act_history = self.act_history[0:self.history_len]
        self.obs_history = self.obs_history[0:self.history_len]

    def aggregate_history(self):
        curr_hist_delta = self.history_len - len(self.obs_history)
        for i in range(curr_hist_delta):
            self.obs_history.append(torch.zeros(self.obs_shape).to(self.device))
        return torch.stack(self.obs_history)

class DQNAgent(BaseAgent):

    def __init__(self,
                 config,
                 device,
                 n_actions,
                 obs_shape):

        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape)
        self.buffer = ReplayMemory(self.buffer_size, "dqn")

    def select_action(self, state):
        sample = np.random.uniform(0, 1)
        eps_threshold = self.eps_final + (self.eps_init - self.eps_final) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.q_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(0, self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return
        transitions = self.buffer.sample(self.batch_size)
        # Transpose the batch
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).float()

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.q_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.t_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # print("loss: ", loss.item())
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

class A2CAgent():
    def __init__(self,
                 config,
                 device,
                 n_actions,
                 obs_shape):

        BaseAgent.__init__(self,
                           **config, 
                           device=device,
                           n_actions=n_actions,
                           obs_shape=obs_shape)
        self.buffer = ReplayMemory(self.buffer_size, "a2c")
        self.actor = Actor(self.obs_size, self.n_actions, self.hidden_size)
        self.actor.to(self.device)

    def update_history(self, act, state):
        self.act_history.insert(0, torch.tensor(act).to(self.device))
        self.obs_history.insert(0,  torch.tensor(state).to(self.device))
        self.act_history = self.act_history[0:self.history_len]
        self.obs_history = self.obs_history[0:self.history_len]

    def aggregate_history(self):
        curr_hist_delta = self.history_len - len(self.obs_history)
        for i in range(curr_hist_delta):
            self.obs_history.append(torch.zeros(self.obs_shape).to(self.device))
        return torch.stack(self.obs_history)

    def select_action(self, state):
        sample = np.random.uniform(0, 1)
        eps_threshold = self.eps_final + (self.eps_init - self.eps_final) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        dist = self.actor(state.flatten())
        
        return dist

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return
        transitions = self.buffer.sample(self.batch_size)
        # Transpose the batch
        batch = A2CTransition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None]).float()
        non_final_index = torch.stack([i for i in range(batch.next_state)
                                              if batch.next_state[i] is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        log_prob_batch = torch.cat(batch.log_prob)
        dist_batch = torch.cat(batch.dist)

        import pdb; pdb.set_trace()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.q_net(state_batch).gather(1, action_batch.reshape(self.batch_size, 1))
        # for advantage calculation
        with torch.no_grad():
            state_action_values_no_grad = self.q_net(state_batch).gather(1, action_batch.reshape(self.batch_size, 1))

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        #Not max anymore, Expectation over the policy instead

        next_state_values[non_final_mask] = self.t_net(non_final_next_states)


        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        value_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.q_net_optimizer.zero_grad()
        value_loss.backward()
        # print("value_loss: ", value_loss.item())
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q_net_optimizer.step()

        # Compute advantage = Q(s_t, a) - V(s_t)
        curr_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            curr_state_action_values = self.t_net(state_batch)
            curr_state_values = torch.bmm(curr_state_action_values.reshape(self.batch_size, 1, self.n_actions), 
                                          action_batch.reshape(self.batch_size, self.n_actions, 1))
        advantage = state_action_values_no_grad - curr_state_values
        policy_loss = (-action_batch * advantage).mean()

        # Optimize the policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()
        
        return policy_loss.item(), value_loss.item()

class NashActorCriticAgent():

    def __init__(self):
        pass