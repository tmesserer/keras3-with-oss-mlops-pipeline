# PRD: ML Training & Serving Pipeline (Learning Project)

## Overview
End-to-end ML pipeline for learning: train, track, serve, monitor, and
collect human feedback on model predictions (fraud detection use case).

## Architecture

### Containers
1. **postgres** — database for Labelbox
2. **mlflow** — experiment tracking + model registry
3. **training** — ephemeral, trains model, logs to MLflow, config passed via env vars
4. **bentoml** — serves champion model from MLflow via FastAPI; samples 1% of
   inferences (async, non-blocking) and logs to Labelbox
5. **labelbox** — self-hosted, open source; stores sampled inferences for
   human review/labeling
6. **airflow** — orchestrates: triggers training runs, polls MLflow for new
   champion, rebuilds/redeploys bentoml container on promotion
7. **test-client** — Flask app; generates 1 synthetic inference request/sec
   to bentoml, logs request + response time to local SQLite (volume-mounted),
   serves a simple read-only dashboard on port 5500

### Data flow
training → logs to mlflow → airflow detects new champion → rebuilds/deploys
bentoml → bentoml serves predictions → 1% sampled to labelbox → humans label
→ feeds next training run

test-client independently generates load against bentoml for demo/observability
purposes only (not part of the training loop).

## Tech stack
- Keras 3 (PyTorch backend)
- MLflow (tracking + registry)
- BentoML (serving, wraps FastAPI/uvicorn)
- Labelbox (open source, self-hosted, human-in-the-loop labeling)
- Airflow (operational orchestration — not GitHub Actions)
- Flask + SQLite (test-client)
- Docker Compose (all environments)

## Repo structure

your-project/
├── PRD.md
├── docker-compose.yml
├── docker-compose.test.yml
├── docker-compose.prod.yml
├── .env.example
├── bentoml_service/
├── mlflow/
├── labelbox/
├── training/
├── test_client/
│ ├── app.py
│ └── data/ # SQLite, volume-mounted
└── airflow/
└── dags/
├── train_model.py
└── deploy_champion.py


## Out of scope (v1)
- Authentication/authorization on Labelbox, MLflow, BentoML
- TLS/SSL
- Secrets manager (using `.env` files for now)
- Network isolation between containers

## Future considerations
- Nginx reverse proxy + basic auth in front of Labelbox and MLflow when
  shared beyond one laptop
- API keys / rate limiting on BentoML
- Move secrets to a vault when deploying to company server