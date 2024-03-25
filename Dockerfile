FROM python:3.11-alpine

WORKDIR /app

COPY requirements_api.txt .
COPY requirements_ai.txt .

RUN pip3 install --upgrade pip
RUN pip install --no-cache-dir -r requirements_ai.txt
RUN pip install --no-cache-dir -r requirements_api.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port", "5000"]