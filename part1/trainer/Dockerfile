FROM python:3.11-slim
# CPU-only torch first (its own index), then everything else from PyPI.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY train.py .
ENV KERAS_BACKEND=torch PYTHONUNBUFFERED=1
CMD ["python", "train.py"]
