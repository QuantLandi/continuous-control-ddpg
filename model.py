import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    
class Actor(nn.Module):
    """Actor (Policy) Model"""

    def __init__(self, state_size, action_size, seed, hidden_layer_units=256):
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
        # why do we reset parameters here? why didn't we do that in DQN?
        self.reset_parameters()

    def reset_parameters(self):
        self.hidden_layer_logits.weight.data.uniform_(*hidden_init(self.hidden_layer_logits))
        self.output_layer_logits.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Builds an actor (policy) network that maps states -> actions"""
        hidden_layer_activation = F.relu(self.hidden_layer_logits(state))
        output_layer_activation = F.tanh(self.output_layer_logits(hidden_layer_activation))
        return output_layer_activation


class Critic(nn.Module):
    """Critic (Value) Model"""

    def __init__(self, state_size, action_size, seed,
                 layer1_units=256, layer2_units=256, layer3_units=128):
        """Initializes parameters and defines model architecture.
        
        Parameters
        ----------
            state_size : int
                Dimension of each state
            action_size : int
                Dimension of each action
            seed : int
                Random seed
            layer1_units : int
                Number of nodes in first hidden layer
            layer2_units : int
                Number of nodes in second hidden layer
            layer3_units : int
                Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # define model architecture
        self.layer1_logits = nn.Linear(state_size, layer1_units)
        self.layer2_logits = nn.Linear(layer1_units+action_size, layer2_units)
        self.layer3_logits = nn.Linear(layer2_units, layer3_units)
        # in DDPG we output only one action because it is an off-policy algorithm,
        # hence action choice is deterministic, not probabilistic
        # the action is continuous, not discrete
        self.output_layer_logit = nn.Linear(layer3_units, 1)
        # why do we reset parameters here? why didn't we do that in DQN?
        self.reset_parameters()

    def reset_parameters(self):
        self.layer1_logits.weight.data.uniform_(*hidden_init(self.layer1_logits))
        self.layer2_logits.weight.data.uniform_(*hidden_init(self.layer2_logits))
        self.layer3_logits.weight.data.uniform_(*hidden_init(self.layer3_logits))
        self.output_layer_logit.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Builds a critic (value) network that maps (state, action) -> Q-value. 
        """
        layer1_activation = F.leaky_relu(self.layer1_logits(state))
        layer1_activation = torch.cat((layer1_activation, action), dim=1)
        layer2_activation = F.leaky_relu(self.layer2_logits(layer1_activation)) 
        layer3_activation = F.leaky_relu(self.layer3_logits(layer2_activation))
        output_layer_logit = self.output_layer_logit(layer3_activation)
        return output_layer_logit
