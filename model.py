import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model"""

    def __init__(self, state_size, action_size, seed, output_layer_units=256):
        """Initializes parameters and defines model architecture.
        
        Parameters
        ----------
            state_size : int
                Dimension of each state
            action_size : int
                Dimension of each action
            seed : int
                Random seed
            hidden_layer_units : int
                Number of nodes in hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # define model architecture
        self.hidden_layer_logits = nn.Linear(state_size, hidden_layer_units)
        self.output_layer_logits = nn.Linear(hidden_layer_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.hidden_layer_logits.weight.data.uniform(*hidden_init(self.layer1_logits))
        self.output_layer_logits.weight.data.uniform(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions"""
        hidden_layer_activation = F.relu(self.hidden_layer_logits(state))
        output_layer_activation = F.tanh(self.output_layer_logits(hidden_layer_activation))
        return output_layer_activation
