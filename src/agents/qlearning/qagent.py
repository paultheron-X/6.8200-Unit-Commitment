import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl4uc import processor

from gym import spaces

# torch detect anomaly
#torch.autograd.set_detect_anomaly(True)

class QNetwork(nn.Module):
    def __init__(self, obs_size, num_nodes, n_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_size, num_nodes),
            nn.ReLU(),
            nn.Linear(num_nodes, n_out),
            #nn.Sigmoid()
        )
        #self.init_network()

    def forward(self, ob):
        return self.layers(ob)
    
    def init_network(self):
        # init everything to 1
        for p in self.parameters():
            p.data.fill_(1)

class QAgent():
    def __init__(self, env, cfg, **kwargs):
        super(QAgent, self).__init__()
        self.__dict__.update(kwargs)
        self.cfg = cfg
        self.num_gen = env.num_gen
        
        self.batch_size = cfg['batch_size']
        self.warmup_steps = cfg['warmup_steps']
        self.forecast_horizon = 12
        self.obs_processor = processor.LimitedHorizonProcessor(env, forecast_horizon=self.forecast_horizon)
        
        self.gamma = cfg['gamma']
        self.tau = cfg['tau']
        
        self.initial_epsilon = cfg['initial_epsilon']
        self.min_epsilon = cfg['min_epsilon']
        self.max_decay_episodes = cfg['max_epsilon_decay_steps']
        
        self.target_update_freq = cfg['target_update_freq']
        self.enable_double_q = (cfg['enable_double_q']=="True")
        
        self.obs_size = self.process_observation(env.reset()).size
        self.num_nodes = self.obs_size
        self.action_size = env.action_size
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_size,), dtype=np.int16)
        
        self.reset()
        print(self.q)
    
    def reset(self):
        self.epsilon = self.initial_epsilon
        self.ep_reduction = (self.epsilon - self.min_epsilon) / float(self.max_decay_episodes)
        
        self.q = QNetwork(self.obs_size, self.num_nodes, 2*self.num_gen)
        self.target_q = QNetwork(self.obs_size, self.num_nodes, 2*self.num_gen)
        
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.cfg['lr'])
        #self.loss_criterion = nn.MSELoss()
        #self.loss_criterion = nn.BCEWithLogitsLoss()
        self.loss_criterion = nn.HuberLoss()
        
        self.q.train()
        self.target_q.train()
        
    def process_observation(self, obs):
        """
        Process an observation into a numpy array.
        
        Observations are given as dictionaries, which is not very convenient
        for function approximation. Here we take just the generator up/down times
        and the timestep.
        
        Customise this!
        """
        obs_new = self.obs_processor.process(obs)
        # obs_new = np.concatenate((obs['status'], [obs['timestep']]))
        return obs_new
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.ep_reduction, self.min_epsilon)
    
    def forward(self, obs):
        return self.q.forward(obs)
    
    #@torch.no_grad()
    def act(self, obs, greedy_only=False, warmup=False):
        """
        Agent always acts greedily w.r.t Q-values!
        """
        processed_obs = self.process_observation(obs)
        processed_obs = torch.from_numpy(processed_obs).float()
        
        if warmup:
            action = self.action_space.sample()
            return action, processed_obs
        
        if greedy_only or (np.random.rand() >= self.epsilon):
            q_values = self.forward(processed_obs)
            q_values = q_values.reshape(self.num_gen, 2)
            action = q_values.argmax(axis=1).detach().numpy()
        else:
            action = self.action_space.sample()
        
        return action, processed_obs
    
    def generate_multiple_actions_batched(self, env, obs, N_samples, threshold, lower_threshold=True):
        """
        Function that generates N actions sequentially.
        """
        # Repeat obs N times
        processed_obs = self.process_observation(obs)

        q_values = self.forward(processed_obs)
        q_values = q_values.reshape(self.num_gen, 2)
        action = q_values.argmax(axis=1).detach().numpy()
        
        q_values_softmax = nn.functional.softmax(q_values, dim=1)
        gaps = q_values_softmax[:,1] - q_values_softmax[:,0]
        candidates = np.where(gaps >= threshold)[0]

        # Create action dictionary
        action_dict = {}
        for idx in candidates:
            action_sub = np.copy(action)
            action_sub[idx] = 1 - action[idx]
            # Convert action to bit string
            action_id = ''.join(str(int(i)) for i in action_sub)
            # Convert bit string to int
            action_id = int(action_id, 2)
            action_dict[action_id] = action_sub

        return action_dict, 0
    
    def update(self, memory, batch_size=None):
        
        #print("--> Before", self.q.layers[0].weight)
        if memory.num_used < self.warmup_steps:
            return 0
        
        if batch_size == None:
            batch_size = self.batch_size
        
        data = memory.sample(batch_size)
        
        self.optimizer.zero_grad()
        
        qs = self.q(torch.as_tensor(data['obs']).float()).float()
        qs = qs.reshape(batch_size, self.num_gen, 2)
        
        #print(type(qs), qs.requires_grad)
        m,n = data['act'].shape
        I,J = np.ogrid[:m,:n]
        qs = qs[I, J, data['act']]
        #print(type(qs), qs.requires_grad)
        
        
        next_qs = self.target_q(torch.as_tensor(data['next_obs']).float()).reshape(batch_size, self.num_gen, 2)
        next_acts = next_qs.argmax(axis=2).detach().numpy()
        m,n = next_acts.shape
        I,J = np.ogrid[:m,:n]
        next_qs = next_qs[I, J, next_acts]
        
        m,n = next_qs.shape
        rews = np.broadcast_to(data['rew'], (self.num_gen,batch_size)).T
        rews = torch.as_tensor(np.copy(rews)).float()

        td_target = rews + self.gamma * next_qs
        
        loss = self.loss_criterion(qs, td_target)
        
        print('state_dict', self.q.state_dict()['layers.2.weight'][0, 0:5])
        
        loss.backward()
        self.optimizer.step()
        
    def update_target(self):
        target_net_dict = self.target_q.state_dict()
        current_net_dict = self.q.state_dict()
        update_dict = {param: self.tau * target_net_dict[param] + (1 - self.tau) * current_net_dict[param] for param in target_net_dict.keys()}
        self.target_q.load_state_dict(update_dict)
        
class ReplayMemory(object):

    def __init__(self, capacity, obs_size, act_dim):
        
        self.capacity = capacity
        self.obs_size = obs_size
        self.act_dim = act_dim 
        
        self.act_buf = np.zeros((self.capacity, self.act_dim))
        self.obs_buf = np.zeros((self.capacity, self.obs_size))
        self.rew_buf = np.zeros(self.capacity)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_size))
        
        self.num_used = 0
        
    def store(self, obs, action, reward, next_obs):
        """Store a transition in the memory"""
        idx = self.num_used % self.capacity
        
        self.act_buf[idx] = action
        self.obs_buf[idx] = obs
        self.rew_buf[idx] = reward
        self.next_obs_buf[idx] = next_obs
        
        self.num_used += 1
    
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.capacity), size=batch_size, replace=False)
        
        data = {'act': self.act_buf[idx],
                'obs': self.obs_buf[idx],
                'rew': self.rew_buf[idx],
                'next_obs': self.next_obs_buf[idx]}
        
        return data
        
    def is_full(self):
        return (self.num_used >= self.capacity)
    
    def reset(self):
        self.num_used = 0 