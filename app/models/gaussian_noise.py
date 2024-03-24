import torch
import torch.nn as nn

class GaussianNoise(n.Module):
	"""
	Neural Network to add Gaussian Noise to a dataset

	Arguments : 
		- mu (optional) : the mean of the noise distribution
		- sigma (optional and > 0) : the standard deviation of the noise distribution
	"""   
 	def __init__(self,mu=0.5,sigma = 0.1):
 		super().__init__()
 		self.sigma = sigma
 		self.mu = mu
 		self.noise = torch.tensor(0).to(device)

 	def forward(self, input):
 		if self.training and self.sigma > 0:
 			# Generate noise which follows gaussian distribution
 			scale = self.sigma * x.detach()
 			noise = self.noise.repeat(*x.size()).normal_(mean = self.mu) * scale
 
 			x += noise # Add noise to x
 		return x