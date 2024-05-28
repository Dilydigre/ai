CONFIG = {
	"debug": True,
	"api_path" : {
		"model_v1": "/api_v1/",		# base route for model 1 API
		"model_v2": "/api_v2/"		# base route for model 2 API
	},
	"routes": {
		"generate_without_prompt": "generate",
		"generate_with_prompt": "generate_prompt"
	},
	"api_image_format":"PNG",
	"time_format":" %d/%m/%Y %H:%M:%S ",

	"model_file_path" : {
		"model_v1": "models/models_weight/modele1.ckpt",		# path of first model ckpt file
		"model_v2": None
	}
}