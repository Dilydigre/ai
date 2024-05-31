import torch
import torch.nn as nn
import torchvision
from .gaussian_noise import GaussianNoise
from .UNET import UNet
import numpy as np

class UNETModel:
	
	def __init__(self, PATH):
		self.gaussian_noise = GaussianNoise(25)
		self.unet = UNet()
		self.model_PATH=PATH
		self.unet.load_state_dict(torch.load(self.model_PATH)['model_state_dict']) #load le fichier de la VM de save/modele1 dans le UNET
		self.unet.eval()
# On crée ensuite les deux fonctions nécessaires à la liaison entre l'AI et l'API 

	def generate_without_prompt(self):
		noise_image = torch.tensor(np.random.rand(1, 3, 512, 512).astype('float32')) #image bruitée
		unet_image = self.unet(noise_image) #image bruitée passée à travers le U-Net
		resized_image = torch.reshape(unet_image, (512, 512, 3)) # on reshape l'image avec torch.reshape
		normalized_image = torch.clamp(resized_image, 0, 1)*255 # on normalise avec torch.clamp pour avoir des valeurs entre 0 et 255
		return normalized_image.detach().numpy().astype("uint8")

	def generate_with_prompt(self, prompt):
		return self.generate_without_prompt()
