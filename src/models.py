import random
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque
from torch.distributions import Normal

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
A2CTransition = namedtuple('Transition',
                           ('state', 'action', 'dist', 'next_state', 'reward'))
NashACTransition = namedtuple('Transition',
                              ('state', 'a', 'b', 'dist', 'next_state', 'reward'))
RfNashACTransition = namedtuple('Transition',
                                ('state', 'a', 'b'))

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

class ReplayMemory(object):

    def __init__(self, capacity, transition_type):
        self.memory = deque([],maxlen=capacity)
        self.transition_type = transition_type

    def push(self, *args):
        """Save a transition"""
        if self.transition_type == "a2c":
            self.memory.append(A2CTransition(*args))
        elif self.transition_type == "nash_ac":
            self.memory.append(NashACTransition(*args))
        elif self.transition_type == "rf_nash_ac":
            self.memory.append(RfNashACTransition(*args))
        else:
            self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ConvModel(nn.Module):

    def __init__(self, in_channels, h, w, out_size):
        super(ConvModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        linear_input_size = self.linear_input_size(in_channels, h, w)

        self.head = nn.Linear(linear_input_size, out_size)

    def linear_input_size(self, in_channels, h, w):
        x = torch.ones((1, in_channels, h, w))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))).flatten()

        return x.shape[0]

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        return self.head(x.view(x.size(0), -1))

class MLPModel(nn.Module):

    def __init__(self, in_size, out_size, hidden_size, num_layers=0, batch_norm=False):
        super(MLPModel, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        self.hidden = []
        self.bns = []
        self.in_layer = nn.Linear(self.in_size, self.hidden_size)
        for i in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_size))
        self.out_layer = nn.Linear(self.hidden_size, self.out_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.in_layer(x))
        for i in range(self.num_layers):
            if self.batch_norm:
                x = F.relu(self.bns[i](self.hidden[i](x)))
            else:
                x = F.relu(self.hidden[i](x))
        return self.out_layer(x)

class Actor(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, temperature=1):
        super(Actor, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.temperature = temperature
        
        self.hidden1 = nn.Linear(in_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, out_size)     

    def forward(self, state):
        x = F.relu(self.hidden1(state), inplace=False)
        return F.softmax(self.out_layer(x)/self.temperature)

class LSTMModel(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, lstm_out, num_layers=1):
        super(LSTMModel, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_out = lstm_out

        self.lstm = nn.LSTM(in_size, lstm_out, num_layers+1, batch_first = True)
        self.out_layer = nn.Linear(self.lstm_out, self.out_size)

    def forward(self, x):
        # h_0 defaults to 0
        x, hidden = self.lstm(x)
        return self.out_layer(x[:,-1,:])

class LinearQHead(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearQHead, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.linear = nn.Linear(self.in_size, self.out_size)

    def forward(self, x):
        return self.linear(x)