import torch
import pandas as pd
import numpy as np

class NewLogger:

	def __init__(self, num_epochs, num_workers, steps_per_epoch, *args):

		self.num_epochs = num_epochs
		self.steps_per_epoch = steps_per_epoch
		self.num_workers = num_workers
		self.log = {}
		for key in args:
			self.log[key] = torch.zeros((num_epochs, num_workers)).share_memory_()

	def store(self, key, value, epoch, worker_id):

		self.log[key][epoch, worker_id] = value

	def save_to_csv(self, fn):

		df = pd.DataFrame()
		for key in self.log:
			df[key] = self.log[key].numpy().mean(axis=1)
		df['epoch'] = np.arange(self.num_epochs)
		df['timestep'] = df['epoch'] * self.steps_per_epoch
		df.to_csv(fn, index=False)

class Logger:

	def __init__(self, num_epochs, num_workers, steps_per_epoch):

		self.num_epochs = num_epochs
		self.steps_per_epoch = steps_per_epoch
		self.num_workers = num_workers
		self.reward = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.std_reward = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.q25_reward = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.q75_reward = torch.zeros((num_epochs, num_workers)).share_memory_()
		
		self.timesteps = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.std_timesteps = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.q25_timesteps = torch.zeros((num_epochs, num_workers)).share_memory_()
		self.q75_timesteps = torch.zeros((num_epochs, num_workers)).share_memory_()



	def log(self, reward, std_reward, q25_reward, q75_reward, 
			timesteps, std_timesteps, q25_timesteps, q75_timesteps,
			epoch, worker_id):

		self.reward[epoch, worker_id] = reward
		self.std_reward[epoch, worker_id] = std_reward
		self.q25_reward[epoch, worker_id] = q25_reward
		self.q75_reward[epoch, worker_id] = q75_reward

		self.timesteps[epoch, worker_id] = timesteps
		self.std_timesteps[epoch, worker_id] = std_timesteps
		self.q25_timesteps[epoch, worker_id] = q25_timesteps
		self.q75_timesteps[epoch, worker_id] = q75_timesteps


	def save_to_csv(self, fn):
		
		# Training epoch and timesteps observed
		epoch = np.arange(self.num_epochs)
		timestep = epoch * self.steps_per_epoch

		avg_reward = self.reward.numpy().mean(axis=1)
		std_reward = self.std_reward.numpy().mean(axis=1)
		q25_reward = self.q25_reward.numpy().mean(axis=1)
		q75_reward = self.q75_reward.numpy().mean(axis=1)
		
		avg_timesteps = self.timesteps.numpy().mean(axis=1)
		std_timesteps = self.std_timesteps.numpy().mean(axis=1)
		q25_timesteps = self.q25_timesteps.numpy().mean(axis=1)
		q75_timesteps = self.q75_timesteps.numpy().mean(axis=1)


		df = pd.DataFrame({'epoch': epoch,
						   'timestep': timestep, 
						   'avg_reward': avg_reward,
						   'std_reward': std_reward,
						   'q25_reward': q25_reward,
						   'q75_reward': q75_reward,
						   'avg_timesteps': avg_timesteps,
						   'std_timesteps': std_timesteps,
						   'q25_timesteps': q25_timesteps,
						   'q75_timesteps': q75_timesteps})

		df.to_csv(fn, index=False)
