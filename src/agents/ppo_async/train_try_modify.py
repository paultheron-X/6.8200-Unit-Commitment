#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import copy
from collections import namedtuple
import pandas as pd 
import os

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR


from agents.ppo_async.ac_agent import ACAgent
from agents import helpers
from agents.agent_training_logger import Logger, NewLogger
from agents.helpers import discount_cumsum

from rl4uc.environment import make_env, make_env_from_json

device = "cpu"
mp.set_start_method('spawn', True)

MsgUpdateRequest = namedtuple('MsgUpdateRequest', ['agent', 'update'])

def run_epoch(save_dir, env, local_ac, shared_ac, pi_optimizer, v_optimizer, epoch_counter):
    obs = env.reset()
    epoch_done = False
    done = False
    ep_reward, ep_timesteps = 0, 0
    ep_rews, ep_vals, ep_sub_acts = [], [], []

    rewards = []
    timesteps = []

    while epoch_done is False:

        # Choose action
        action, sub_obs, sub_acts, log_probs = local_ac.generate_action(env, obs)
        
        # Get state-value pair
        value, obs_processed = local_ac.get_value(obs)
        
        # Advance environment
        new_obs, reward, done, _ = env.step(action)

        # Simple transformation of reward
        # reward = 1+reward/-env.min_reward
        reward = reward / -env.min_reward
        reward = np.log(-1/reward)

        # Clip to between (-10, 10)
        reward = reward.clip(-10, 10)

        # Update episode rewards and timesteps survived
        ep_reward += reward
        ep_rews.append(reward)
        ep_vals.append(value.detach().item())
        ep_sub_acts.append(len(sub_acts))
        ep_timesteps += 1

        local_ac.critic_buffer.store(obs_processed, reward)
        for idx in range(len(sub_obs)):
            local_ac.actor_buffer.store(sub_obs[idx], sub_acts[idx], log_probs[idx], reward, value)

        obs = new_obs

        if done:
            local_ac.actor_buffer.finish_ep_new(ts=ep_sub_acts, 
                                                ep_rews=ep_rews,
                                                ep_vals=ep_vals,
                                                last_val=0)
            local_ac.critic_buffer.finish_ep(last_val=0)
            
            rewards.append(ep_reward)
            timesteps.append(ep_timesteps)

            obs = env.reset()
            ep_reward, ep_timesteps = 0,0
            ep_rews, ep_vals, ep_sub_acts = [], [], []

        if local_ac.actor_buffer.is_full():
            if not done: 
                local_ac.actor_buffer.finish_ep_new(ts=ep_sub_acts, 
                                                    ep_rews=ep_rews,
                                                    ep_vals=ep_vals,
                                                    last_val=local_ac.get_value(obs)[0].detach().numpy())
                local_ac.critic_buffer.finish_ep(last_val=local_ac.get_value(obs)[0].detach().numpy())

            entropy, loss_v, explained_variance = shared_ac.update(local_ac, pi_optimizer, v_optimizer)
            mean_entropy, loss_v, explained_variance = torch.mean(entropy).item(), loss_v.item(), explained_variance.item()

            epoch_done = True

        done = False
            
    log(save_dir, rewards, timesteps, mean_entropy, loss_v, explained_variance)
    if epoch_counter % EPOCH_SAVE_INTERVAL == 0:
        print("---------------------------")
        print("saving actor critic weights")
        print("---------------------------")
        save_ac(save_dir, shared_ac, epoch_counter) 

def run_worker(save_dir, rank, num_epochs, shared_ac, epoch_counter, env_params, params):
    """
    Training with a single worker. 
    
    Each worker initialises its own optimiser. Parameters for the policy network
    are shared between workers.
        
    Results are written to .txt files which are shared between workers.
    """
    start_time = time.time()
    
    pi_optimizer = optim.Adam(shared_ac.parameters(), lr=params.get('ac_learning_rate'))
    v_optimizer = optim.Adam(shared_ac.parameters(), lr=params.get('cr_learning_rate'))
    

    np.random.seed(params.get('seed') + rank)
    env = make_env(**env_params)
    
    local_ac = ACAgent(env, **params)
        
    while epoch_counter < num_epochs:
        
        epoch_counter += 1 
        if epoch_counter % 1000 == 0:
            print("Epoch: {}".format(epoch_counter.item()))

        # If using target entropy, then schedule
        if params.get('entropy_target', 0) != 0:
            shared_ac.entropy_coef = params.get('entropy_coef') * (3 * epoch_counter.item() / num_epochs )

        # Anneal entropy
        ANNEAL_ENTROPY = False
        if ANNEAL_ENTROPY: 
            new_coef = (1 - epoch_counter / num_epochs) * params['entropy_coef']
            print(new_coef)
            local_ac.entropy_coef = shared_ac.entropy_coef = new_coef

        
        local_ac.load_state_dict(shared_ac.state_dict())
                
        # Run an epoch, including updating the shared network
        run_epoch(save_dir, env, local_ac, shared_ac, pi_optimizer, v_optimizer, epoch_counter)
    
    # Record time taken
    time_taken = time.time() - start_time
    with open(os.path.join(save_dir, 'time_taken.txt'), 'w') as f:
        f.write(str(time_taken) + '\n')
        
def train(save_dir,
		timesteps, 
	    num_workers, 
		steps_per_epoch, 
		env_params, 
		policy_params,
        args
    ):
    
    epoch_counter = torch.tensor([0])
    epoch_counter.share_memory_()
    num_epochs = int(timesteps / steps_per_epoch)
    
    # initialise environment and the shared networks 
    # env = make_env(**env_params)
    env = make_env_from_json(args.env_name)
    policy = ACAgent(env, **policy_params)

    if args.ac_weights_fn is not None:
        print("********************Using pre-trained AC weights***********************")
        policy.load_state_dict(torch.load(args.ac_weights_fn))

    policy.train()
    policy.share_memory()

	# Total number of epochs (updates)

	# Number of timesteps each worker should gather per epoch
    worker_steps_per_epoch = int(steps_per_epoch / num_workers)

    #pi_optimizer = optim.Adam(policy.parameters(), lr=policy_params.get('ac_learning_rate'))
    #v_optimizer = optim.Adam(policy.parameters(), lr=policy_params.get('cr_learning_rate'))

    # The actor buffer will typically take more entries than the critic buffer,
    # because it records sub-actions. Hence there is usually more than one entry
    # per timestep. Here we set the size to be the max possible.

    log_keys = ('mean_reward', 'std_reward', 'q25_reward', 'q75_reward',
        'mean_timesteps', 'std_timesteps', 'q25_timesteps', 'q75_timesteps', 'entropy')
    logger = NewLogger(num_epochs, num_workers, steps_per_epoch, *log_keys)


    # Worker update requests
    update_request = [False]*num_workers
    epoch_counter = 0

    workers = []
    pipes = []

    for worker_id in range(num_workers):
        p_start, p_end = mp.Pipe()
        worker = Worker(worker_id=str(worker_id),env=env,policy=policy,pipe=p_end,logger=logger,num_epochs=num_epochs,steps_per_epoch=worker_steps_per_epoch)
        worker.start()
        workers.append(worker)
        pipes.append(p_start)

    start_time = time.time()

	# starting training loop
    while epoch_counter < num_epochs:
        for i, conn in enumerate(pipes):
            if conn.poll():
                msg = conn.recv()

                # if agent is waiting for network update
                if type(msg).__name__ == "MsgUpdateRequest":
                    update_request[i] = True
                    if False not in update_request:

                        print("Epoch: {}".format(epoch_counter))
                        print("Updating")


                        entropy, loss_v, explained_variance = policy.update(actor_buf, critic_buf, pi_optimizer, v_optimizer)
                        print("Entropy: {}".format(entropy.mean()))
                        for w in range(num_workers):
                            logger.store('entropy', entropy.mean().detach(), epoch_counter, w)

                        epoch_counter += 1
                        update_request = [False]*num_workers
                        msg = epoch_counter

                        # periodically save the logs
                        if (epoch_counter + 1) % 10 == 0:
                            logger.save_to_csv(os.path.join(save_dir, 'logs.csv'))

						# send to signal subprocesses to continue
                        for pipe in pipes:
                            pipe.send(msg)

    time_taken = time.time() - start_time

    logger.save_to_csv(os.path.join(save_dir, 'logs.csv'))
    torch.save(policy.state_dict(), os.path.join(save_dir, 'ac_final.pt'))

    # Record training time
    with open(os.path.join(save_dir, 'time_taken.txt'), 'w') as f:
        f.write(str(time_taken) + '\n')
        
        
class Worker(mp.Process):
    def __init__(self, worker_id, env, pipe, policy, gamma, lam, 
                num_epochs, steps_per_epoch, logger):

        mp.Process.__init__(self, name=worker_id)

        self.worker_id = worker_id
        self.policy = policy
        self.env = copy.deepcopy(env)
        self.gamma = gamma
        self.lam = lam
        self.pipe = pipe
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.logger = logger
  
    def run(self):
        for epoch in range(self.num_epochs):
            all_ep_rewards, all_ep_timesteps = self.run_epoch()

            self.logger.store('mean_reward', np.mean(all_ep_rewards), epoch, int(self.worker_id))
            self.logger.store('std_reward', np.std(all_ep_rewards), epoch, int(self.worker_id))
            self.logger.store('q25_reward', np.quantile(all_ep_rewards, 0.25), epoch, int(self.worker_id))
            self.logger.store('q75_reward', np.quantile(all_ep_rewards, 0.75), epoch, int(self.worker_id))
            self.logger.store('mean_timesteps', np.mean(all_ep_timesteps), epoch, int(self.worker_id))
            self.logger.store('std_timesteps', np.std(all_ep_timesteps), epoch, int(self.worker_id))
            self.logger.store('q25_timesteps', np.quantile(all_ep_timesteps, 0.25), epoch, int(self.worker_id))
            self.logger.store('q75_timesteps', np.quantile(all_ep_timesteps, 0.75), epoch, int(self.worker_id))

            msg = MsgUpdateRequest(int(self.worker_id), True)
            self.pipe.send(msg)
            msg = self.pipe.recv()
   
    def run_epoch(self):
        pass


   
        
if __name__ == "__main__":
    
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--workers', type=int, required=False, default=1)
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--buffer_size', type=int, required=False, default=2000)
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1000000))
    parser.add_argument('--num_gen', type=int, required=True)
    parser.add_argument('--timesteps', type=int, required=True)
    parser.add_argument('--steps_per_epoch', type=int, required=True)


    # The following params will be used to setup the PPO agent
    parser.add_argument('--ac_learning_rate', type=float, required=False, default=3e-05)
    parser.add_argument('--cr_learning_rate', type=float, required=False, default=3e-04)
    parser.add_argument('--ac_arch', type=str, required=False, default="32,32,32")
    parser.add_argument('--cr_arch', type=str, required=False, default="32,32,32")
    parser.add_argument('--entropy_coef', type=float, required=False, default=0.01)
    parser.add_argument('--clip_ratio', type=float, required=False, default=0.1)
    parser.add_argument('--forecast_horizon_hrs', type=int, required=False, default=12)
    parser.add_argument('--credit_assignment_1hr', type=float, required=False, default=0.9)
    parser.add_argument('--minibatch_size', type=int, required=False, default=None)
    parser.add_argument('--update_epochs', type=int, required=False, default=4)
    parser.add_argument('--observation_processor', type=str, required=False, default='LimitedHorizonProcessor')
    parser.add_argument('--gradient_steps', type=int, required=False, default=10)
    parser.add_argument('--entropy_target', type=float, required=False, default=None)

    # Alternatively, pass a filename for trained weights and parameters, used to set network architectures.
    parser.add_argument('--ac_weights_fn', type=str, required=False, default=None)
    parser.add_argument('--ac_params_fn', type=str, required=False, default=None)
    
    args = parser.parse_args()

    # Make save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Load policy params and save them to the local directory. 
    policy_params = vars(args)
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(policy_params, sort_keys=True, indent=4))
    
    # Load the env params and save them to save_dir
    env_params = helpers.retrieve_env_params(args.num_gen)
    with open(os.path.join(args.save_dir, 'env_params.json'), 'w') as fp:
        fp.write(json.dumps(env_params, sort_keys=True, indent=4))
    
    # If training using a pre-defined AC networks --> overwrite archs
    if args.ac_params_fn is not None:
        ac_params = json.load(open(args.ac_params_fn))
        policy_params.update({'ac_arch': ac_params['ac_arch'], 'cr_arch': ac_params['cr_arch']})
    
    # Check if cuda is available:
    if torch.cuda.is_available():
         torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train(save_dir=args.save_dir,
		timesteps=args.timesteps,
		num_workers=args.workers,
		steps_per_epoch=args.steps_per_epoch,
		env_params=env_params,
		policy_params=policy_params,
        args = policy_params)