# Documentation
- [API documentation](#API)
- [AI documentation](#AI)


## API
### Online documentation
If API is running, complete documentation with examples can be found at `<app_adress>/docs`
### Offline documentation
> *Last update : 08/03/2024*
#### Root API
| path | methods | parameters | return | details |
|---|---|---|---|---|
| / | GET,POST | - | status as boolean in "status" | if false, error while setting up API, if true API runs |
#### First model API
> Root path : /api_v1/

| path | methods | parameters | return | details |
|---|---|---|---|---|
| / | GET,POST|  - | status as boolean in "status" | if false, error while loading model, if true model is loaded correctly |
| /status | GET,POST| - | status as boolean in "status" | same as / |
| /generate | GET,POST | - | status as boolean in "status" and base64 encoded png image in "image" | status = false means error during image generation, image will then be none|
| /generate_prompt | POST | "description" as str | status as boolean in "status" and base64 encoded png image in "image" | status = false means error during image generation, image will then be none|

