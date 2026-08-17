# Same rule as app/: no ML dependency. This service moves rows between Postgres and Label
# Studio's REST API and never touches a model, so it stays small and starts instantly.
FROM python:3.11-slim
COPY labeling/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY labeling/sync.py labeling/label_config.xml pipeline/schema.sql ./
ENV PYTHONUNBUFFERED=1
EXPOSE 8002
CMD ["uvicorn", "sync:app", "--host", "0.0.0.0", "--port", "8002"]
