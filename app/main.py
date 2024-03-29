from fastapi import FastAPI
from .routers import model_v1, response_model
from .config import CONFIG

app = FastAPI()

app.include_router(model_v1.router.router)

@app.get("/", response_model=response_model.StatusResponse)
async def root() -> response_model.StatusResponse:
	"""
	Return the status of the API
	"""
	return {"status" : True}