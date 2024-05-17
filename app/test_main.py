from fastapi.testclient import TestClient
from base64 import b64decode

from .main import app
from .config import CONFIG
from .util import isBase64, isImage

# Intialize client to test API
client = TestClient(app)

# Load routes from config file
API_1_ROUTE = CONFIG['api_path']["model_v1"]
API_2_ROUTE = CONFIG['api_path']["model_v2"]
SUB_PATH = CONFIG['routes']


# Global route
def test_root_get():
	response = client.get("/")
	assert response.status_code == 200
	assert response.json() == {"status" : True}

def test_root_post():
	response = client.post("/")
	assert response.status_code == 200
	assert response.json() == {"status" : True}



# Syntax for test functions of models : test_[model name]_<path name>_[method]

###########################
### Tests for first API ###
###########################
# => Root path
def test_model1_get():
	response = client.get(API_1_ROUTE)
	assert response.status_code == 200
	assert response.json() == {"status" : True}

def test_model1_post():
	response = client.post(API_1_ROUTE)
	assert response.status_code == 200
	assert response.json() == {"status" : True}

# => Path for generation without any user prompt
def test_model1_generate_get():
	response = client.get(API_1_ROUTE+SUB_PATH["generate_without_prompt"])
	assert response.status_code == 200
	assert response.json()["status"]
	assert isBase64(response.json()["image"])
	assert isImage(b64decode(response.json()["image"]))

def test_model1_generate_post():
	response = client.post(API_1_ROUTE+SUB_PATH["generate_without_prompt"])
	assert response.status_code == 200
	assert response.json()["status"]
	assert isBase64(response.json()["image"])
	assert isImage(b64decode(response.json()["image"]))

# => Path for generation with user prompt
def test_model1_generate_prompt_get(): # GET is not allowed for this path, response code should be 405
	response = client.get(API_1_ROUTE+SUB_PATH["generate_with_prompt"])
	assert response.status_code == 405

def test_model1_generate_prompt_post():
	response = client.post(API_1_ROUTE+SUB_PATH["generate_with_prompt"])
	assert response.status_code == 200
	assert response.json()["status"]
	assert isBase64(response.json()["image"])
	assert isImage(b64decode(response.json()["image"]))

############################
### Tests for second API ###
############################
# => Root path
def test_model2_get():
	response = client.get(API_2_ROUTE)
	assert response.status_code == 200
	assert response.json() == {"status" : True}

def test_model2_post():
	response = client.post(API_2_ROUTE)
	assert response.status_code == 200
	assert response.json() == {"status" : True}

# => Path for generation without any user prompt
def test_model2_generate_get():
	response = client.get(API_2_ROUTE+SUB_PATH["generate_without_prompt"])
	assert response.status_code == 200
	assert response.json()["status"]
	assert isBase64(response.json()["image"])
	assert isImage(b64decode(response.json()["image"]))

def test_model2_generate_post():
	response = client.post(API_2_ROUTE+SUB_PATH["generate_without_prompt"])
	assert response.status_code == 200
	assert response.json()["status"]
	assert isBase64(response.json()["image"])
	assert isImage(b64decode(response.json()["image"]))

# => Path for generation with user prompt
def test_model2_generate_prompt_get(): # GET is not allowed for this path, response code should be 405
	response = client.get(API_2_ROUTE+SUB_PATH["generate_with_prompt"])
	assert response.status_code == 405

def test_model2_generate_prompt_post():
	response = client.post(API_2_ROUTE+SUB_PATH["generate_with_prompt"])
	assert response.status_code == 200
	assert response.json()["status"]
	assert isBase64(response.json()["image"])
	assert isImage(b64decode(response.json()["image"]))

