import random
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
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
        return self.softmax(self.out_layer(x))
