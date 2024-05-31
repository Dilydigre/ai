FROM python:3.10
WORKDIR /app

COPY requirements_api.txt .
COPY requirements_ai.txt .

RUN apt update -y && apt install git wget cargo -y

RUN pip  install --upgrade pip && pip install nvidia-pyindex && pip  install --no-cache-dir --no-deps -r requirements_ai.txt && pip  install --no-cache-dir -r requirements_api.txt

COPY . .

RUN wget -q -O app/models/models_weight/ld-model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

RUN wget --user-agent Mozilla/4.0 https://dl-m6jmv1ql.swisstransfer.com/api/download/6f06f95f-476d-4a44-8054-36b811a0495d/57516a41-10bf-4f19-8a03-2ebaa9675daa -O app/models/models_weight/modele1.ckpt

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port", "5000"]
