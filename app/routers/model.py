from app.config import CONFIG
from app.util import print_message
from .response_model import *
from typing import *
from fastapi import APIRouter

class ModelAPI:
	def __init__(self,api_base_url:str, model_status: bool) -> None:
		self.api_base_url = api_base_url
		self.router = APIRouter()
		self.model_status = model_status
		self.add_api_route("status", self.get_status, methods=['GET','POST'], response_model = StatusResponse)
		self.add_api_route("", self.get_status, methods=['GET','POST'], response_model = StatusResponse)

	def add_api_route(self,route: str,function: Callable,methods: List[str], response_model: Any = None) -> None:
		if CONFIG['debug']:
			print_message("INFO", "Add api route at "+self.api_base_url+route, "ModelAPI class")

		self.router.add_api_route(self.api_base_url + route, function, methods=methods, response_model=response_model)

	def get_status(self) -> StatusResponse:
		"""
		Return the status of the loaded model
		"""
		return {"status" : self.model_status}