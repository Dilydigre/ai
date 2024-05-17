from typing import *
import time as tm
import base64
from PIL import Image
import io

from .config import CONFIG

def print_message(level: str,message: str,emitter: str):
	print("[i] ["+tm.strftime(CONFIG['time_format'],tm.gmtime())+"] "+level.upper()+"\t"+emitter+"\t"+message)

def print_error(error: Exception,emitter: str):
	print("[!] ["+tm.strftime(CONFIG['time_format'],tm.gmtime())+"] ERROR\t"+emitter+"\t"+str(error))

def isBase64(s: str):
	try:
		base64.b64decode(s, validate=true)
	except:
		return False
	return True

def isImage(byte_image: bytes):
	try:
		Image.open(io.BytesIO(byte_image))
	except:
		return False
	return True
