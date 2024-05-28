import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
	"""
	Neural Network to add Gaussian Noise to a dataset

	Arguments : 
		- step : number of layers of gaussian noise
		- mu (optional) : the mean of the noise distribution
		- sigma (optional and > 0) : the standard deviation of the noise distribution
	"""   
	def __init__(self,step,mu=0.5,sigma = 0.1, device="cpu"):
		super().__init__()
		self.step = step
		self.sigma = sigma
		self.mu = mu
		self.noise = torch.tensor(0).float()

	def forward(self, x):
		if self.training and self.sigma > 0:
			for i in range(self.step):
			# Generate noise which follows gaussian distribution
				scale = self.sigma * x.detach()
				noise = self.noise.repeat(*x.size()).normal_(mean = self.mu) * scale
	
				x += noise # Add noise to x
		return x