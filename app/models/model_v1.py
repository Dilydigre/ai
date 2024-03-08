from app.util import print_message,print_error

import numpy as np

# temporary model to test api
class Model:
	def __init__(self):
		print_message("INFO","Init model...","Model v1")
	def generate_without_prompt(self):
		return (np.random.rand(100,100,3) * 255).astype('uint8')

	def generate_with_prompt(self,prompt):
		return (np.random.rand(200,200,3) * 255).astype('uint8')