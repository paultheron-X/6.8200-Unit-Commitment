import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from rl4uc.environment import make_env_from_json

from qagent import QAgent, ReplayMemory

def train(save_dir, env_name, nb_epochs):
    
    MEMORY_SIZE = 200
    N_EPOCHS = nb_epochs
    
    env = make_env_from_json(env_name)
    agent = QAgent(env)
    memory = ReplayMemory(MEMORY_SIZE, agent.obs_size, env.num_gen)
    
    log = {'mean_timesteps': [],
           'mean_reward': []}
    
    for i in range(N_EPOCHS):
        if i % 10 == 0:
            print("Epoch {}".format(i))
        epoch_timesteps = []
        epoch_rewards = []
        while memory.is_full() == False:
            done = False
            obs = env.reset()
            timesteps = 0
            while not done: 
                action, processed_obs = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                
                if not done:
                    next_obs_processed = agent.process_observation(next_obs)
                    memory.store(processed_obs, action, reward, next_obs_processed)
                
                obs = next_obs
                
                if memory.is_full():
                    break
                
                timesteps += 1
                if done:
                    epoch_rewards.append(reward)
                    epoch_timesteps.append(timesteps)
                    
        log['mean_timesteps'].append(np.mean(epoch_timesteps))
        log['mean_reward'].append(np.mean(epoch_rewards))
        
        agent.update(memory)
        memory.reset()
                    
    return agent, log

if __name__ == "__main__":
	
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Train Q-Learning agent')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--nb_epochs', type=int, required=True)

    args = parser.parse_args()

    agent, log = train(
        save_dir = args.save_dir,
        env_name = args.env_name,
        nb_epochs = args.nb_epochs
        )
    
    pd.Series(log['mean_reward']).rolling(50).mean().plot()
    
    
 