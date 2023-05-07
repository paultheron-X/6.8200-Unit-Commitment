import numpy as np
from gym import spaces

from rl4uc import processor

class MAB():
    def __init__(self, env, cfg, **kwargs):
        super(MAB, self).__init__()
        self.__dict__.update(kwargs)
        self.cfg = cfg
        self.num_gen = env.num_gen
        self.action_size = env.action_size
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_size,), dtype=np.int16)
        self.initial_epsilon = cfg['initial_epsilon']
        self.min_epsilon = cfg['min_epsilon']
        self.max_decay_episodes = cfg['max_epsilon_decay_steps']
        self.reset()

    def reset(self):
        self.epsilon = self.initial_epsilon
        self.ep_reduction = (self.epsilon - self.min_epsilon) / float(self.max_decay_episodes)
        self.action_counts = np.zeros(2**self.action_size) 
        self.Q = np.zeros(2**self.action_size) 

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.ep_reduction, self.min_epsilon)

    def update(self, action, reward):
        self.action_counts[action] += 1
        self.Q[action] += (1.0 / self.action_counts[action]) * (reward - self.Q[action])
        
    def act(self, obs):
        """
        Agent always acts greedily w.r.t Q-values!
        """
        if np.random.rand() >= self.epsilon:
            idx = np.argmax(self.Q)
            action = self.index_to_action(idx, self.action_size)
        else:
            action = self.action_space.sample()
            idx = self.action_to_index(action)
        return action, idx
    
    def index_to_action(self, num, m): # convert index to associated action
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)
    
    def action_to_index(self, action): # convert action to associated index
        return int("".join(str(x) for x in action), 2)
   