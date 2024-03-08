from pydantic import BaseModel

class RequestPrompt(BaseModel):
    description: str

class ImageResponse(BaseModel):
	status: bool
	image: str | None = None

class StatusResponse(BaseModel):
	status: bool
