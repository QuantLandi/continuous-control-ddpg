import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

BUFFER_SIZE = 1000000  # replay buffer size
BATCH_SIZE = 128       # minibatch size
GAMMA = 0.99           # discount factor
TAU = 0.001            # target parameter soft update rate
LR_ACTOR = 0.0001      # actor learning rate
LR_CRITIC = 0.0003     # critic learning rate
WEIGHT_DECAY = 0.0001  # critic gradient descent optimizer weight decay rate

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Parameters
        ----------
            state_size : int
                Dimension of each state
            action_size : int
                Dimension of each action
            random_seed : int
                Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)
        
        # create actor's local and target networks and local network's optimizer
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        
        # create critic's local and target networks and local network's optimizer
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.actor_critic.parameters(),
                                           lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # create Ornstein-Uhlenbeck noise to be added to action space
        self.noise = OUNoise(action_size, random_seed)
        
        # create replay buffer
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE,
                                          BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, is_episode_over):
        """
        Saves experience tuple in replay buffer and
        samples batch of experience tuples at random.
        """
        # add (s, a, r, s') experience tuple to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, is_episode_over)
        # if enough samples are available in replay buffer, learn from experience
        enough_samples = len(self.replay_buffer) > BATCH_SIZE
        if enough_samples:
            experiences = self.replay_buffer.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        # set local actor network in evaluation mode
        self.actor_local.eval()
        # temporarily deactivate PyTorch autograd engine
        # to reduce memory usage and speed up computation
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        # set local actor network in training mode
        self.actor_local.train()
        # add temporally correlated noise to explore well
        # in a physical environment with momentum
        if add_noise:
            action += self.noise.sample()
        action = np.clip(action, -1, 1)
        return action
