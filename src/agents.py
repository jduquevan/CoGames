import torch
import torch.nn.functional as F
import numpy as np

from functools import reduce

class DQNAgent():

    def __init__(self, 
                 n_actions, 
                 hidden_size, 
                 num_layers, 
                 batch_norm, 
                 obs_shape, 
                 buffer_size, 
                 model_type):

        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.obs_shape = obs_shape
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