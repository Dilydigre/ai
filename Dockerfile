FROM python:3.11-alpine

WORKDIR /app

COPY requirements_api.txt .
COPY requirements_ai.txt .

RUN pip3 install --upgrade pip
RUN pip install --no-cache-dir -r requirements_api.txt
RUN pip install --no-cache-dir -r requirements_api.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host","127.0.0.1", "--port", "5000"]