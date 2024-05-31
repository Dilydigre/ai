FROM python:3.10
WORKDIR /app

COPY requirements_api.txt .
COPY requirements_ai.txt .

RUN apt update -y && apt install git wget cargo -y

RUN pip  install --upgrade pip && pip install nvidia-pyindex && pip  install --no-cache-dir --no-deps -r requirements_ai.txt && pip  install --no-cache-dir -r requirements_api.txt

COPY . .

RUN wget -q -O app/models/models_weight/ld-model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

RUN wget --user-agent Mozilla/4.0 'https://download.wetransfer.com/eugv/cd55cf1c3cc108046f4123681309a49a20240531174839/9223d435abb211117fc4d17b5e4bb1db1ec5a149/modele1.ckpt?cf=y&token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6ImRlZmF1bHQifQ.eyJleHAiOjE3MTcxNzgzNjQsImlhdCI6MTcxNzE3Nzc2NCwiZG93bmxvYWRfaWQiOiJiYjU5MGJkNS1iNWFiLTQ5ZjMtYTdiYS03ZGM2MTY0MjdmYzYiLCJzdG9yYWdlX3NlcnZpY2UiOiJzdG9ybSJ9.bgA4UataZo6pbM3QsydF0qozn017VwNbDVH45WbE2yc' -O models/models_weight/modele1.ckpt

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port", "5000"]
