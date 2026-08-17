FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow>=3.15,<4"
WORKDIR /mlruns
EXPOSE 5000
# command comes from docker-compose.yml so you can tweak it without rebuilding
