from app.util import print_message, print_error

import numpy as np

# temporary model to test api which returns random image

class Model:
	def __init__(self):
		print_message("INFO","Init model...","Model v1")

	def generate_without_prompt(self):
		return (np.random.rand(512, 512, 3) * 255).astype('uint8')

	def generate_with_prompt(self, prompt):
		return (np.random.rand(200, 200, 3) * 255).astype('uint8')