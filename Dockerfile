FROM python:lastest-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["uvicorn","app.main:app","--host","127.0.0.1","--port","5000"]