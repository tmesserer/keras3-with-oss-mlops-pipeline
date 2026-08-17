# No torch, no keras, no features.py. This app consumes decisions over HTTP and knows
# nothing about the model -- which is why the image is ~100MB instead of ~2GB. If this
# Dockerfile ever needs an ML dependency, the serving boundary has leaked.
FROM python:3.11-slim
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY app/app.py app/index.html pipeline/schema.sql ./
ENV PYTHONUNBUFFERED=1
EXPOSE 5500
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5500"]
