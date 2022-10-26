import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import reduce

from models import MLPModel, ReplayMemory, Transition

class DQNAgent():

    def __init__(self,
                 device,
                 n_actions,
                 eps_init,
                 eps_final,
                 eps_decay,
                 hidden_size,
                 num_layers,
                 batch_norm, 
                 obs_shape,
                 buffer_size, 
                 model_type,
                 batch_size):
        self.steps_done = 0
        self.device = device
        self.n_actions = n_actions
        self.eps_init = eps_init
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.obs_shape = obs_shape
        self.batch_size = batch_size
        self.obs_size = reduce(lambda a, b: a * b, self.obs_shape)

        self.buffer = ReplayMemory(buffer_size)

        if model_type.lower() == "mlp":
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
        else:
            raise Exception("Not supported model type for DQN Agent")
        
        self.t_net.load_state_dict(self.q_net.state_dict())

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

    def optimize_model(self, optimizer):
        if len(self.buffer) < self.batch_size:
            return
        transitions = self.buffer.sample(self.batch_size)
        # Transpose the batch
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
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
        optimizer.zero_grad()
        loss.backward()
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    def target(self):
        """Returns the target Q-values for states/observations."""
        pass

    def step(self):
        """Compute the discrete distribution for the Q-value for each
        action for each state/observation (no grad)."""
        pass

class NashActorCriticAgent():

    def __init__(self):
        pass

    def target(self):
        """Returns the target Q-values for states/observations."""
        pass

    def step(self):
        """Compute the discrete distribution for the Q-value for each
        action for each state/observation (no grad)."""
        pass