FROM python:3.10
WORKDIR /app

COPY requirements_api.txt .
COPY requirements_ai.txt .

RUN apt update -y && apt install git wget cargo -y

RUN pip  install --upgrade pip && pip install nvidia-pyindex && pip  install --no-cache-dir --no-deps -r requirements_ai.txt && pip  install --no-cache-dir -r requirements_api.txt

COPY . .

RUN wget -q -O app/models/models_weight/ld-model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

RUN mkdir repos
WORKDIR repos

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install git-lfs && git lfs install && git clone https://github.com/Dilydigre/ai.git

WORKDIR ../

RUN cp repos/ai/app/models/models_weight/modele1.ckpt app/models/models_weight/modele1.ckpt && rm -rf repos

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port", "5000"]