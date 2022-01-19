import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size):
		self.max_size = int(max_size)
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.weight = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device


		self.state_mean = None
		self.state_std = None


	def standardizer(self, state):
		if self.state_mean is None:
			self.state_mean = np.mean(self.state, axis=0, keepdims=True)
			self.state_std = np.std(self.state, axis=0, keepdims=True) + 1e-3
			self.state_mean = torch.FloatTensor(self.state_mean).to(self.device)
			self.state_std = torch.FloatTensor(self.state_std).to(self.device)
		
		return (state - self.state_mean) / self.state_std

	def unstandardizer(self, state):
		return state * self.state_std + self.state_mean

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def normalize_states(self, eps=1e-3):
		mean = self.state.mean(0, keepdims=True)
		std = self.state.std(0, keepdims=True) + eps
		self.state = (self.state - mean) / std
		self.next_state = (self.next_state - mean) / std
		return mean, std


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.size])
		np.save(f"{save_folder}_action.npy", self.action[:self.size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]

	
	def load_buffers(self, save_folder1, save_folder2, size = -1):
		reward_buffer1 = np.load(f"{save_folder1}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer1.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder1}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder1}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder1}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer1[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder1}_not_done.npy")[:self.size]

		print(self.size)

		reward_buffer2 = np.load(f"{save_folder2}_reward.npy")
		online_buffer_size = reward_buffer2.shape[0]
		self.state[self.size:] = np.load(f"{save_folder2}_state.npy")[:online_buffer_size]
		self.action[self.size:] = np.load(f"{save_folder2}_action.npy")[:online_buffer_size]
		self.next_state[self.size:] = np.load(f"{save_folder2}_next_state.npy")[:online_buffer_size]
		self.reward[self.size:] = reward_buffer2[:online_buffer_size]
		self.not_done[self.size:] = np.load(f"{save_folder2}_not_done.npy")[:online_buffer_size]

		self.size += online_buffer_size
		print(self.size)

