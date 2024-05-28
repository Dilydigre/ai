from app.routers.model import ModelAPI
from app.config import CONFIG
from app.util import print_message,print_error
from .response_model import *

from typing import *
from PIL import Image
import io
import base64 as b64
import time as tm
import traceback

EMITTER = "API model_v1" # Usefull for debug

async def generate_face_without_prompt() -> ImageResponse:
	"""
	Description : ask the model to generate face without any prompt\n
	Return : the status of the generation and the associated base64 encoded png image generated by the model
	"""
	if model is not None:
		try:
			generated_array = model.generate_without_prompt() # generate image using model
			generated_png = Image.fromarray(generated_array) # convert to png

			buffer = io.BytesIO() 										  # see https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save and
			generated_png.save(buffer, format=CONFIG['api_image_format']) # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.tobytes for the method

			return {
				"status" : True,
				"image" : b64.b64encode(buffer.getbuffer())
			} # return status and base64 encoded raw png image
		
		except Exception as e:
			
			if CONFIG['debug']:	# error message if in debug mode
				print_error(e,EMITTER)
				traceback.print_exc()
	return {"status":False,"image":None}


async def generate_face_with_prompt(prompt: RequestPrompt) -> ImageResponse:
	"""
	Description : ask the model to generate face without a prompt\n
	Return : the status of the generation and the associated base64 encoded png image generated by the model
	"""
	if model is not None:
		try:

			generated_array = model.generate_with_prompt(prompt)
			generated_png = Image.fromarray(generated_array) # convert to png

			buffer = io.BytesIO() 										  # see https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save and
			generated_png.save(buffer, format=CONFIG['api_image_format']) # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.tobytes for the method

			return {
				"status" : True,
				"image" : b64.b64encode(buffer.getbuffer())
			} # return status and base64 encoded raw png image
		
		except Exception as e:
			if CONFIG['debug']:	# error message if in debug mode
				print_error(e,EMITTER)
				traceback.print_exc()

	return {
		"status" : False,
		"image":None
	}

# Try to load model
model = None
try:
	from app.models.UNET_model import UNETModel
	model = UNETModel(CONFIG['model_file_path']['model_v1'])
except Exception as e:
	# If loading fails, print error if debug and set status to "false" i.e not ready
	if CONFIG['debug']:
		print_error(e,EMITTER)
		traceback.print_exc()

# setup router
router = ModelAPI(CONFIG['api_path']['model_v1'], model is not None)
	
# add routes
router.add_api_route(
	CONFIG['routes']['generate_without_prompt'],
	generate_face_without_prompt,['GET','POST'],
	response_model=ImageResponse
)
router.add_api_route(
	CONFIG['routes']['generate_with_prompt'],
	generate_face_with_prompt, ['POST'],
	response_model=ImageResponse
)