import torch
import torch.nn as nn
import torchvision
from .gaussian_noise import GaussianNoise
from .UNET import UNet

class UNETModel:
	
	def __init__(self, PATH):
		self.gaussian_noise = GaussianNoise(1)
		self.unet = UNet()
		self.model_PATH=PATH
		self.unet.load_state_dict(torch.load(self.model_PATH)) #load le fichier de la VM de save/modele1 dans le UNET

# On crée ensuite les deux fonctions nécessaires à la liaison entre l'AI et l'API 

	def generate_without_prompt(self):
		#tensor 3 512 512 avec tous les coefficients à 0 
		noise_image = self.gaussian_noise() #image bruitée
		unet_image = self.unet(noise_image) #image bruitée passée à travers le U-Net
		resized_image = torch.reshape(unet_image, (512, 512, 3)) # on reshape l'image avec torch.reshape
		normalized_image = torch.clamp(resized_image, 0, 1)*255 # on normalise avec torch.clamp pour avoir des valeurs entre 0 et 255
		return normalized_image

	def generate_with_prompt(self, prompt):
		return self.generate_without_prompt()
