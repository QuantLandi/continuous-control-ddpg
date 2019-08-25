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
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
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

    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        """
        Updates policy and value parameters using batch of experience tuples.

        Sets Q_targets = r + (gamma * critic_target(next_states, next_actions))
        where next_actions = actor_target(next_states))

        Parameters
        ----------
            experiences : tuple
                Batch of experience tuples (s, a, r, s', is_episode_over)
                sampled from replay buffer
            gamma : float
                Discount factor for value of next state
        """
        states, actions, rewards, next_states, is_episode_over = experiences
        
        # update critic
        # predict next actions
        next_actions = self.actor_target(next_states)
        # predict Q targets for next states
        next_Q_targets = self.critic_target(next_states, next_actions)
        # set next_Q_targets = 0 where state is terminal 
        # so that Q_targets = rewards, if state is terminal
        next_Q_targets = next_Q_targets * (1 - is_episode_over)
        # compute y_i, i.e. Q targets for current states
        Q_targets = rewards + (gamma * next_Q_targets)
        Q_predictions = self.critic_local(states, actions)
        # compute critic prediction loss with respect to targets
        critic_loss = F.mse_loss(Q_predictions, Q_targets)
        # minimize loss
        # clear gradients
        self.critic_optimizer.zero_grad()
        # compute gradients
        critic_loss.backward()
        # update parameters
        self.critic_optimizer.step()
        
        # update actor
        # compute actor loss
        predicted_actions = self.actor_local(states)
        # since we aim at maximizing the value of (s, a) under policy mu,
        # we minimize its negative        
        actor_loss = -self.critic_local(states, predicted_actions).mean()
        # minimize loss
        # clear gradients
        self.actor_optimizer.zero_grad()
        # compute gradients
        actor_loss.backward()
        # update parameters
        self.actor_optimizer.step()

        # update target weights
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(local_network, target_network, tau):
        """
        Performs soft update of network weights.
        Sets target_weights = tau*local_weights + (1-tau)*target_weights

        Parameters
        ----------
            local_network : PyTorch neural network
                Local network to copy weights from
            target_network : PyTorch neural network
                Target network to copy weights to
            tau : float
                Interpolation parameter
        """
        local_weights_batch = local_network.parameters()
        target_weights_batch = target_network.parameters()
        for local_weights, target_weights in zip(local_weights_batch,
                                                 target_weights_batch):
            updated_weights = tau*local_weights.data + (1-tau)*target_weights.data
            target_weights.data.copy_(updated_weights)

class OUNoise:
    """Ornstein-Uhlenbeck process"""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Resets internal state (i.e. noise) to mean (i.e. mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Updates internal state and return it as noise sample."""
        x = self.state
        noise = np.random.rand(len(x))
        dx = self.theta * (self.mu - x) + self.sigma * noise
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Stores experience tuples to be sampled by the agent."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initializes a ReplayBuffer object.

        Parameters
        ----------
            buffer_size : int
                Maximum number of experience tuples to store in replay buffer
            batch_size : int
                Number of experience tuples to be collected from replay buffer

        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        field_names = ['state', 'action', 'reward', 'next_state', 'is_episode_over']
        self.experience = namedtuple('Experience', field_names=field_names)
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, is_episode_over):
        """Adds new experience batch to replay buffer memory."""
        experience = self.experience(state, action, reward, next_state, is_episode_over)
        self.memory.append(experience)

    def sample(self):
        """Samples batch of experiences at random from replay memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [e.state for e in experiences if e is not None]
        actions = [e.action for e in experiences if e is not None]
        rewards = [e.reward for e in experiences if e is not None]
        next_states = [e.next_state for e in experiences if e is not None]
        is_episode_over = [e.is_episode_over for e in experiences if e is not None]

        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        is_episode_over = np.vstack(is_episode_over).astype(np.uint8)
        is_episode_over = torch.from_numpy(is_episode_over).float().to(device)

        return (states, actions, rewards, next_states, is_episode_over)
