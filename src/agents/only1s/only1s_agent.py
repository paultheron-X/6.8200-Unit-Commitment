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
    
    def generate_action(self, **kwargs):
        """
        Agent always acts greedily w.r.t Q-values!
        """
        action = np.ones(self.action_size)
        return action
    
    def generate_multiple_actions_batched(self,**kwargs):
        """
        Agent always acts greedily w.r.t Q-values!
        """
        
        n_action = 1/0.05
        actions =  np.ones((n_action,self.action_size))
        
        action_dict = {}
        for i in range(n_action):
            action_dict[i] = actions[i]
        return action_dict, 0
   