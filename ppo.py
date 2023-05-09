from network import FeedForwardNN

class PPO:
	def __init__(self, env):
		# Extract environment information
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		# Initialize actor and critic networks
		self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
		self.critic = FeedForwardNN(self.obs_dim, 1)
		self._init_hyperparameters()

	def _init_hyperparameters(self):
		# Default values for hyperparameters, will need to change later.
		self.timesteps_per_batch = 4800            # timesteps per batch
		self.max_timesteps_per_episode = 1600      # timesteps per episode