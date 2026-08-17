# Keras 3 + MLflow

A small MLOps loop — **train → compare → promote → serve → observe** — split across three
containers that share nothing but the MLflow registry.

| container | role | lifecycle |
|---|---|---|
| `mlflow-server` | tracking + model registry (sqlite + file artifacts) | long running, :5000 |
| `trainer` | Keras 3 on the torch backend, one run per launch | launched per run, exits |
| `api-serve` | FastAPI + uvicorn serving `models:/mnist-keras@champion` | you start it, :8000 |

The split is the point: the trainer is ephemeral and configured entirely from outside, the
server outlives it, and `api-serve` has never seen `train.py` — it finds the model through the
registry alias alone.

Host folders mounted into the containers:

```
mlruns/      MLflow backend store + artifacts (mounted at /mlruns in every container)
data/        cached mnist.npz                           -> /data
inference/   predictions.jsonl, appended per request    -> /inference
```

## Run it (WSL or Linux)

```bash
cd keras3-with-mlflow
sed -i "s/^DOCKER_UID=.*/DOCKER_UID=$(id -u)/;s/^DOCKER_GID=.*/DOCKER_GID=$(id -g)/" .env

python3.12 -m venv .venv && source .venv/bin/activate   # 3.12, not 3.14: pyarrow has no cp314 wheels
pip install -r requirements-host.txt

jupyter lab notebooks/hub.ipynb     # or open the folder in VS Code and pick .venv as kernel
```

Then work through the notebook top to bottom. Section 3 trains three models, one cell each;
change a number in any cell and re-run it — nothing is rebuilt, because the hyperparameters
live outside the image.

The host venv only holds the MLflow client, pandas and requests; Keras and torch live inside
the containers.

## Manual equivalents

```bash
docker compose up -d mlflow-server                       # http://localhost:5000

docker compose --profile tools run --rm \
  -e RUN_NAME=cnn-32 -e KIND=cnn -e SIZE=32 -e LR=1e-3 trainer

docker compose --profile serve up -d api-serve           # http://localhost:8000/health
curl localhost:8000/predict/sample/7
tail -f inference/predictions.jsonl
docker compose --profile tools --profile serve down
```

## Notes

- **Champion selection**: notebook step 5 sets the `champion` alias on the best registered
  version. `api-serve` resolves the alias at startup; `POST /reload` re-resolves it live.
- **Where models live**: only in the MLflow registry, as the `keras` flavor. There is
  deliberately no second copy on the host — the registry is the source of truth.
- **Artifacts**: `./mlruns` is mounted at the same absolute path in all containers, so
  `file:///mlruns/artifacts` URIs resolve identically from trainer and server.
- **Parameters**: the trainer reads its whole config from env vars (`RUN_NAME`, `KIND`, `SIZE`,
  `DROPOUT`, `OPT`, `LR`, `EPOCHS`, `BATCH_SIZE`, `TRAIN_SUBSET`). `docker-compose.yml` holds
  the defaults; each launch overrides them with `-e`. The image never needs rebuilding to try a
  new configuration.
- **Speed**: defaults are `EPOCHS=3`, `TRAIN_SUBSET=20000` (CPU friendly). Set `TRAIN_SUBSET=0`
  for the full 60k training set.
- **Backend**: `KERAS_BACKEND=torch` everywhere. Swap it to `tensorflow` or `jax` in
  `docker-compose.yml` and the same code trains and serves unchanged.
