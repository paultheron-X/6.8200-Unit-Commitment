import os
import json
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rl4uc.environment import make_env_from_json

from agents.random.random_agent import RandomAgent

def train(save_dir, env_name, verbose=True):

    cfg_folder = os.path.join('src/agents/random/configs/random.json')
    with open(cfg_folder) as f:
        cfg = json.load(f)

    env = make_env_from_json(env_name)
    agent = RandomAgent(env)

    ep_timesteps = []
    ep_rewards = []
    smooth_ep_ret = deque(maxlen=500)
    obs = env.reset()
    nb_ep = 0
    ep_ret = 0
    for t in range(cfg['max_steps']):
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)

        ep_ret += reward
        obs = next_obs

        if done:
            obs = env.reset()
            smooth_ep_ret.append(ep_ret)
            ep_rewards.append(np.mean(smooth_ep_ret))
            ep_timesteps.append(t)
            nb_ep += 1
            ep_ret = 0
            if verbose and nb_ep % 100 == 0:
                print(f'Step {t}, episode {nb_ep}, smoothed reward {ep_rewards[-1]}', end='\r')
    log = {
        'mean_timesteps': ep_timesteps,
        'mean_reward': ep_rewards
    }
         
    return agent, log

if __name__ == "__main__":
	
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Train Random Agent')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--env_fn', type=str, required=True)

    args = parser.parse_args()

    random_agent, log = train(
        save_dir = args.save_dir,
        env_name = args.env_name,
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

    file_name = 'rolling_mean_rewards_random_agent.png'
    plt.savefig(f'{args.save_dir}/{file_name}', format='png')
    plt.close()
    
    # save logs
    env_params = json.load(open(args.env_fn + '.json', 'r'))
    with open(os.path.join(args.save_dir, 'env_params.json'), 'w') as f:
        f.write(json.dumps(env_params, sort_keys=True, indent=4))
    with open(args.save_dir + '/log.json', 'w') as f:
        json.dump(log, f)
    
    
 