from typing import *
import time as tm

from .config import CONFIG

def print_message(level: str,message: str,emitter: str):
	print("[i] ["+tm.strftime(CONFIG['time_format'],tm.gmtime())+"] "+level.upper()+"\t"+emitter+"\t"+message)

def print_error(error: Exception,emitter: str):
	print("[!] ["+tm.strftime(CONFIG['time_format'],tm.gmtime())+"] ERROR\t"+emitter+"\t"+str(error))