import torch
import torch.nn.functional as F
import numpy as np

class NashACAgent():

    def __init__(self):
        pass

    def target(self):
        """Returns the target Q-values for states/observations."""
        pass

    def step(self):
        """Compute the discrete distribution for the Q-value for each
        action for each state/observation (no grad)."""
        pass