import numpy as np
from gym import spaces

class Only1sAgent():
    def __init__(self, env, **kwargs):
        super(Only1sAgent, self).__init__()
        self.__dict__.update(kwargs)
        self.num_gen = env.num_gen
        self.action_size = env.action_size

    def act(self, obs):
        """
        Agent always acts greedily w.r.t Q-values!
        """
        action = np.ones(self.action_size)
        return action
   