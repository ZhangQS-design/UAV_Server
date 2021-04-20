import multiprocessing
from server.InferenceUtils.replay_memory import Memory
from server.InferenceUtils.torch import *
import math
import time


class DroneAgent:

	def __init__(self, policy,valuenet, device, custom_reward=None,
				 mean_action=False, render=False, running_state=None, num_threads=1):
		self.policy = policy
		self.valuenet = valuenet
		self.device = device
		self.custom_reward = custom_reward
		self.mean_action = mean_action
		self.running_state = running_state
		self.render = render
		self.num_threads = num_threads
		return


	def predict(self, state):
		if self.running_state is not None:
			state = self.running_state(state)
		state_var = tensor(state).unsqueeze(0)
		with torch.no_grad():
			if self.mean_action:
				action = self.policy(state_var)[0][0].numpy()
			else:
				action = self.policy.select_action(state_var)[0].numpy()
			action = int(action) if self.policy.is_disc_action else action.astype(np.float64)
		return action

	def predictTakeValue(self, state):
		if self.running_state is not None:
			state = self.running_state(state)
		state_var = tensor(state).unsqueeze(0)
		with torch.no_grad():
			if self.mean_action:
				action = self.policy(state_var)[0][0].numpy()
			else:
				action = self.policy.select_action(state_var)[0].numpy()
			value = self.valuenet.forward(state_var)
			action = int(action) if self.policy.is_disc_action else action.astype(np.float64)
		return action,value








