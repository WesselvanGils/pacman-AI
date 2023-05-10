import os
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn

class FeedForwardNN(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(FeedForwardNN, self).__init__()
		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = F.sigmoid(self.layer3(activation2))

		return output

	def save(self, file_name="model.pth"):
		model_folder_path = "./models"
		if not os.path.exists(model_folder_path):
			os.makedirs(model_folder_path)

		file_name = os.path.join(model_folder_path, file_name)
		torch.save(self.state_dict(), file_name)