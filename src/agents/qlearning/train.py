import os
import json
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from rl4uc.environment import make_env_from_json

from qagent import QAgent, ReplayMemory

def train(**kwargs):
    
    MEMORY_SIZE = 200
    N_EPOCHS = 1000
    
    env = make_env_from_json('5gen')
    agent = QAgent(env)
    memory = ReplayMemory(MEMORY_SIZE, agent.obs_size, env.num_gen)
    
    log = {'mean_timesteps': [],
           'mean_reward': [],
            'smooth_ep_ret': deque(maxlen=100)}
    log['smooth_ep_ret'].append(0)
    log['mean_reward'].append(0)
    log['mean_timesteps'].append(0)
    
    for i in range(N_EPOCHS):
        #print('\n')
        if (i+1) % 10 == 0:
            print("\nEpoch {}".format(i+1))
        epoch_timesteps = []
        epoch_rewards = []
        # print the reward on the same line
        print(f"Reward: {log['mean_reward'][-1]}", end='\r')
        while memory.is_full() == False:
            done = False
            obs = env.reset()
            timesteps = 0
            while not done: 
                action, processed_obs = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                if not done: # otherwise bug because obs lists are empty
                    next_obs_processed = agent.process_observation(next_obs)
                    memory.store(processed_obs, action, reward, next_obs_processed)
                
                obs = next_obs
                
                
                if memory.is_full():
                    break
                
                timesteps += 1
                if done:
                    epoch_rewards.append(reward)
                    epoch_timesteps.append(timesteps)
                    log['smooth_ep_ret'].append(reward)     
        
        log['mean_timesteps'].append(np.mean(epoch_timesteps))
        log['mean_reward'].append(np.mean(epoch_rewards))
        
        agent.update(memory)
        memory.reset()
    
    log['mean_reward'] = log['mean_reward'][1:]
    log['mean_timesteps'] = log['mean_timesteps'][1:]
                    
    return agent, log

if __name__ == "__main__":
	
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Train Q-Learning agent')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--env_fn', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()

    qagent, log = train(
        save_dir = args.save_dir,
        env_name = args.env_name,
        cfg_path = args.config
        )
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    log_rewards = pd.Series(log['mean_reward'])
    window_size = 50
    rolling_mean_rewards = log_rewards.rolling(window=window_size).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(rolling_mean_rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Rolling nean of Mean Rewards')
    plt.title('Rolling Mean of Log Mean Rewards with Window Size {}'.format(window_size))

    file_name = 'rolling_mean_rewards_qagent.png'
    plt.savefig(f'{args.save_dir}/{file_name}', format='png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(qagent.losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss by number of epochs')

    file_name = 'losses_qagent.png'
    plt.savefig(f'{args.save_dir}/{file_name}', format='png')
    plt.close()
    
    # save logs
    env_params = json.load(open(args.env_fn))
    with open(os.path.join(args.save_dir, 'env_params.json'), 'w') as f:
        f.write(json.dumps(env_params, sort_keys=True, indent=4))
    with open(args.save_dir + '/log.json', 'w') as f:
        json.dump(log, f)
        
    # save agent 
    torch.save(qagent.q.state_dict(), os.path.join(args.save_dir, 'qagent_final.pt'))
    
    
 