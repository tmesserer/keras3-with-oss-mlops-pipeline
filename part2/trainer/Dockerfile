# Build context is part2/, not part2/trainer/, so that common/features.py can be copied in.
# The serving image does the same -- one file, one definition, no training/serving skew.
FROM python:3.11-slim
# CPU-only torch first (its own index), then everything else from PyPI.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY trainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY common/features.py trainer/train.py ./
ENV KERAS_BACKEND=torch PYTHONUNBUFFERED=1
CMD ["python", "train.py"]
