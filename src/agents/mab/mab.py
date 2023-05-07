import numpy as np
from gym import spaces

class RandomAgent():
    def __init__(self, env, **kwargs):
        super(RandomAgent, self).__init__()
        self.__dict__.update(kwargs)
        self.num_gen = env.num_gen
        self.action_size = env.action_size
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_size,), dtype=np.int16)
        
    def act(self, obs):
        """
        Agent always acts greedily w.r.t Q-values!
        """
        action = self.action_space.sample()
        return action
   