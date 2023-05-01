import gym
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from rl4uc.environment import make_env

@dataclass
class QLearningAgent:
    env: gym.Env
    learning_rate: float
    gamma: float
    initial_epsilon: float
    min_epsilon: float
    max_decay_episodes: int
    init_q_value: float = 0.

    def __post_init__(self):
        self.num_states = self.env.observation_space.n
        self.reset()

    def decay_epsilon(self):
        ### TODO: decay epsilon, called after every episode. ################
        self.epsilon = max(self.epsilon - self.ep_reduction, self.min_epsilon)
        #####################################################################
    
    def reset(self):
        self.epsilon = self.initial_epsilon
        self.ep_reduction = (self.epsilon - self.min_epsilon) / float(self.max_decay_episodes)
        self.Q = np.ones((self.num_states, self.env.action_space.n)) * self.init_q_value

    def update_Q(self, state, next_state, action, reward, done):
        ### TODO: update self.Q given new experience. #######################
        ### Remember to use done. What should self.Q be when done is True? ###
        if done:
            self.Q[state, action] = reward
        else:
            self.Q[state, action] = (1-self.learning_rate) * self.Q[state, action] + self.learning_rate*(reward + gamma*np.max(self.Q[next_state,:]))
        #####################################################################

    def get_action(self, state):
        ### TODO: select an action given self.Q and self.epsilon ############
        if random.random() < self.epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        #####################################################################
        return action
    
@dataclass
class QLearningEngine:
    env: gym.Env
    agent: Any
    max_episodes: int
    
    def run(self, n_runs=1):
        rewards = []
        log = []
        for i in tqdm(range(n_runs), desc='Runs'):
            sample_complexity = 0
            ep_rewards = []
            self.agent.reset()
            # we plot the smoothed return values
            smooth_ep_return = deque(maxlen=100)
            for t in tqdm(range(self.max_episodes), desc='Episode'):
                state = self.env.reset()
                ret = 0
                while True:
                    action = self.agent.get_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    sample_complexity += 1
                    true_done = done and not info.get('TimeLimit.truncated', False)
                    self.agent.update_Q(state, next_state, action, reward, true_done)
                    ret += reward
                    state = next_state
                    if done:
                        break
                self.agent.decay_epsilon()
                smooth_ep_return.append(ret)
                ep_rewards.append(np.mean(smooth_ep_return))
            rewards.append(ep_rewards)
            run_log = pd.DataFrame({'return': ep_rewards,  
                                    'episode': np.arange(len(ep_rewards)), 
                                    'iqv': self.agent.init_q_value})
            log.append(run_log)
            print(f'Sample Complexity: {sample_complexity}')
        return log

def qlearning_sweep(init_q_values, n_runs=4, max_episodes=60000, epsilon=0.8, learning_rate=0.8):
    logs = dict()
    pbar = tqdm(init_q_values)
    agents = []
    for iqv in pbar:
        pbar.set_description(f'Initial q value:{iqv}')
        env=gym.make('Deterministic-8x8-FrozenLake-v0')
        agent = QLearningAgent(env=env,
                               learning_rate=learning_rate,
                               gamma=0.99,
                               initial_epsilon=epsilon,
                               min_epsilon=0.0,
                               max_decay_episodes=max_episodes,
                               init_q_value=iqv)
        engine = QLearningEngine(env=env, agent=agent, max_episodes=max_episodes)
        ep_log = engine.run(n_runs)
        ep_log = pd.concat(ep_log, ignore_index=True)
        logs[f'{iqv}'] = ep_log

        agents.append(agent)
    logs = pd.concat(logs, ignore_index=True)
    return logs, agents