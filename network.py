import os
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn

class FeedForwardNN(nn.Module):
	def __init__(self, in_dim, hidden_dim, hidden_count, out_dim):
		super(FeedForwardNN, self).__init__()
		self.linear_relu_stack = nn.Sequential()

		self.linear_relu_stack.add_module(f"linear{0}", nn.Linear(in_dim, hidden_dim))
		self.linear_relu_stack.add_module(f"relu{0}", nn.ReLU())

		for i in range(1, hidden_count):
			self.linear_relu_stack.add_module(f"ff{i}", nn.Linear(hidden_dim, hidden_dim))
			self.linear_relu_stack.add_module(f"relu{i}", nn.ReLU())

		self.linear_relu_stack.add_module(f"linear{hidden_count}", nn.Linear(hidden_dim, out_dim))

	def forward(self, obs):
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		output = self.linear_relu_stack(obs)
		return output

	def save(self, file_name="model.pth"):
		model_folder_path = "./models"
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)

		file_name = os.path.join(model_folder_path, file_name)
		torch.save(self.state_dict(), file_name)