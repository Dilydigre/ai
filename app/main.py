from fastapi import FastAPI
from .routers import model_v1
from .routers import model_v2
from .routers import response_model
from .config import CONFIG

app = FastAPI()

app.include_router(model_v1.router.router) # router for model 1
app.include_router(model_v2.router.router) # router for model 2

@app.get("/", response_model=response_model.StatusResponse)
async def root() -> response_model.StatusResponse:
	"""
	Return the status of the API
	"""
	return {"status" : True}