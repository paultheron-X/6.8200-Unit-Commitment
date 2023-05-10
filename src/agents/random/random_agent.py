import numpy as np
from gym import spaces

class RandomAgent():
    def __init__(self, env, **kwargs):
        super(RandomAgent, self).__init__()
        self.__dict__.update(kwargs)
        self.num_gen = env.num_gen
        self.action_size = env.action_size
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_size,), dtype=np.int16)

    def generate_action(self, **kwargs):
        """
        Agent always acts greedily w.r.t Q-values!
        """
        action = self.action_space.sample()
        return action
    
    def generate_multiple_actions_batched(self,**kwargs):
        """
        Agent always acts greedily w.r.t Q-values!
        """
        
        n_action = 1/0.05
        actions = self.action_space.sample((n_action,))
        
        action_dict = {}
        for i in range(n_action):
            action_dict[i] = actions[i]
        return action_dict, 0
        
   