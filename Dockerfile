FROM python:3.10
WORKDIR /app

COPY requirements_api.txt .
COPY requirements_ai.txt .

RUN pip3 install --upgrade pip
RUN pip install --no-cache-dir --no-deps -r requirements_ai.txt
RUN pip install --no-cache-dir -r requirements_api.txt

COPY . .

RUN wget -O app/models/models_weight/ld-model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port", "5000"]