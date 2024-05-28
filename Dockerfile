FROM python:3.11-alpine
WORKDIR /app

COPY requirements_api.txt .
COPY requirements_ai.txt .

RUN apk update && apk add git wget cargo

RUN pip  install --upgrade pip
RUN pip install nvidia-pyindex
RUN pip  install --no-cache-dir --no-deps -r requirements_ai.txt
RUN pip  install --no-cache-dir -r requirements_api.txt

COPY . .

RUN wget -q -O app/models/models_weight/ld-model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

RUN mkdir repos
WORKDIR repos

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apk install git-lfs
RUN git lfs install
RUN git clone https://github.com/Dilydigre/ai.git

WORKDIR ../

RUN cp repos/ai/app/models/models_weight/modele1.ckpt app/models/models_weight/modele1.ckpt
RUN rm -rf repos


EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port", "5000"]