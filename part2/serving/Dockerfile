# Build context is part2/, not part2/serving/, so that common/features.py can be copied in.
# The trainer image does the same -- one file, one definition, no training/serving skew.
FROM python:3.11-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY common/features.py serving/app.py pipeline/schema.sql ./
ENV KERAS_BACKEND=torch PYTHONUNBUFFERED=1
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
