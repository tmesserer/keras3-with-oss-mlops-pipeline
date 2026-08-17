# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning project for MLOps, not a production system.

**Part 1** (`part1/`, complete and frozen) is a **train → compare → promote → serve → observe**
loop on MNIST, using Keras 3 on the torch backend with MLflow as tracking server and registry.

**Part 2** (`part2/`, in progress) moves to fraud detection on the Sparkov credit-card dataset
and adds the things Part 1 can't teach: an application that depends on the inference endpoint,
a human review queue feeding labels back, and a versioned derived dataset. Both parts share one
MLflow registry and one Postgres.

`PRD.md` is the original sketch of Part 2 and is **partly wrong** — it names Labelbox as
"open source, self-hosted", which does not exist (Labelbox is SaaS-only; the project uses Label
Studio CE instead), and it assumes accuracy-based model selection, which is meaningless at a
0.5% fraud rate. Where PRD and code disagree, the code is the truth.

## Commands

Everything runs through Docker Compose. There is no test suite, no linter, and no build system
beyond the images themselves.

`.env` sets `COMPOSE_FILE=docker-compose.yml:compose.part1.yml`, which is what lets a bare
`docker compose` see every service without `-f` flags. **If `.env` is missing, only the base
file loads and `trainer`/`api-serve` appear not to exist.** Create it with
`cp .env.example .env` then set `DOCKER_UID`/`DOCKER_GID` to your own ids.

```bash
docker compose --profile tools --profile serve build

docker compose up -d mlflow-server                    # http://localhost:5000

# One training run. Every hyperparameter is an env override — the image is never rebuilt.
docker compose --profile tools run --rm \
  -e RUN_NAME=cnn-32 -e KIND=cnn -e SIZE=32 -e LR=1e-3 -e EPOCHS=5 trainer

docker compose --profile serve up -d api-serve        # http://localhost:8000/health
curl localhost:8000/predict/sample/7
curl -X POST localhost:8000/reload                    # re-resolve @champion without restarting

docker compose --profile tools --profile serve down
```

The host-side workflow is `part1/notebooks/hub.ipynb`, which drives all of the above via
`!docker compose` and the MLflow client. Its first cell walks up to the repo root (the folder
holding `docker-compose.yml`) and `chdir`s there, so every `!docker compose` line resolves
regardless of where the notebook was launched from. Host venv must be **Python 3.12** (pyarrow ships no cp314
wheels) and holds only the MLflow client, pandas, numpy, requests, jupyterlab — Keras and torch
live exclusively inside the containers.

## Architecture

Compose is split by concern: `docker-compose.yml` holds shared infrastructure, and one overlay
per part holds that part's services.

| service | file | profile | role |
|---|---|---|---|
| `mlflow-server` | base | *(none — always up)* | tracking + registry, sqlite backend, file artifacts, :5000 |
| `postgres` | base | `db`, `part2` | Label Studio store, analyst verdicts, app telemetry |
| `trainer` | part1 | `tools` | one MNIST run per launch, then exits |
| `api-serve` | part1 | `serve` | FastAPI serving `models:/mnist-keras@champion`, :8000 |
| `fraud-trainer` | part2 | `part2`, `tools2` | one fraud run per launch, then exits |
| `fraud-serve` | part2 | `part2`, `serve2` | `models:/fraud-keras@champion` → decisions, :8001 |
| `txn-app` | part2 | `part2`, `app` | replays transactions at the endpoint, dashboard :5500 |
| `label-studio` | part2 | `part2`, `label` | Label Studio CE review UI, Postgres-backed, :8080 |
| `label-sync` | part2 | `part2`, `label` | review band → tasks, verdicts → Postgres, :8002 |
| `airflow` | part2 | `part2`, `orchestrate` | two DAGs wrapping the hand-run steps, :8090 |

Part 2 additions worth knowing before editing it:

- **Both Part 2 images build from context `./part2`**, not their own subdirectory, so each can
  `COPY common/features.py`. That one file is imported by trainer and server alike — changing
  it changes both, which is the point (no training/serving skew) and the hazard (a feature
  edit invalidates every registered model).
- **The operating threshold belongs to the model, not the server.** `train.py` picks it on the
  validation tail and writes it as an `operating_threshold` **run** tag (not a model-version
  tag — read it via `client.get_run(mv.run_id).data.tags`). `fraud-serve` refuses to start if
  it is missing and no `DECLINE_THRESHOLD` override is set; do not add a 0.5 fallback, which at
  a 0.4% base rate would decline a large slice of legitimate traffic.
- **Postgres schema is applied at service startup** from `part2/pipeline/schema.sql`, all
  `CREATE ... IF NOT EXISTS`. There is no migration tool; keep it idempotent.
- **`scored_transactions.trans_num` is UNIQUE and scoring upserts on it**, so replaying the
  same test rows updates in place rather than duplicating.
- **`txn-app` must never gain an ML dependency.** No keras, no torch, no `features.py` — it
  forwards raw rows over HTTP and consumes decisions. That constraint is the point of the
  service (and keeps the image ~100MB); needing one means the serving boundary has leaked.
- **Two telemetry tables, on purpose.** `scored_transactions` is what the server did;
  `app_requests` is what the client experienced. Don't merge them — rows where
  `outcome != 'scored'` have no server-side counterpart, and those are the interesting ones.
- **`label-sync` carries the same no-ML-dependency rule as `txn-app`.** It moves rows between
  Postgres and Label Studio's REST API. The haversine in `sync.py` is duplicated on purpose —
  importing `features.py` would give the queue an opinion about the model's inputs, and it
  only needs a number for a human to read.
- **The labeling config must not show the model's score.** `part2/labeling/label_config.xml`
  renders transaction fields only. Adding probability or decision to the task data turns the
  queue into a confirmation exercise and the labels it produces carry no new information.
- **Verdicts arrive twice, and that's intended.** The webhook is the fast path; `POST
  /pull` re-reads annotations from the API and is the one that recovers deliveries lost to
  downtime. Both upsert on `verdicts.scored_id`, so neither can double-count.
- **Label Studio rejects bare hostnames in webhook URLs** (Django's `URLValidator`), so
  `label-sync` carries the network alias `label-sync.local` purely to satisfy it. `postgres`
  also lists the `label` profile, because compose refuses to start a service whose
  `depends_on` target is not in an active profile.
- **The derived snapshot is built on the host, not in a container.** `dvc` needs the git repo,
  so `part2/pipeline/build_dataset.py` runs from the host venv like `fetch_data.py`. The
  trainer only *reads* the parquet and parses the `.dvc` pointer by hand — never add dvc to an
  image.
- **`.gitignore` uses `data/*`, not `data/`.** Git cannot re-include a file whose parent
  directory is excluded, and `!data/derived/*.dvc` has to work. Changing that line back to
  `data/` silently stops the snapshot pointer from being committable.
- **Rebuilds must be byte-reproducible** — rows are sorted stably by
  `(trans_date_trans_time, trans_num)` and written with `index=False` so unchanged data yields
  the same md5. Anything that makes the parquet churn (an unstable sort, a timestamp column,
  writing the index) destroys the point of versioning it.
- **The trainer withholds reviewed transactions from the test set.** Analyst rows come from
  the `fraudTest.csv` period, so training on them and evaluating on the full test file leaks.
  `test_rows_withheld` records how many; don't remove the filter to make numbers look better.
- **Airflow launches sibling containers, it does not train.** `fraud_retrain`'s `train` task is
  a `DockerOperator` running `keras-mlops-fraud-trainer:latest` through the host docker socket,
  so the training environment stays defined in one image. The cost is that the trainer's env
  block is restated in the DAG (compose cannot export a service's environment to a sibling) —
  `compose.part2.yml` is the source of truth; a knob added there needs adding in
  `TRAINER_ENV` too. The socket needs `DOCKER_SOCK_GID` (`stat -c '%g' /var/run/docker.sock`).
  Hyperparameters in `TRAINER_ENV` are Jinja (`dag_run.conf` over `params`), which is what
  makes `--conf '{"epochs": "15"}'` work — `environment` is a templated field.
- **Bind sources in the DAG are HOST paths**, from `HOST_REPO_DIR: "${PWD}"`. The daemon
  resolves them on the host, so airflow's own `/data` mount is useless to the container it
  starts. This is also why compose must be run from the repo root.
- **`build_dataset.py` is not a task, on purpose.** The DAG only *verifies* that the parquet's
  md5 still equals the committed `.dvc` pointer and fails with the rebuild command otherwise.
  A pipeline that rewrote the pointer nobody reviewed would version data by accident, and dvc
  would have to enter an image. That is the third hand-rolled copy of the pointer parser
  (`build_dataset.py`, `train.py`, `fraud_retrain.py`) — each in an image that must not import
  the others.
- **Promotion is a comparison, and it can legitimately skip.** A run that does not beat the
  incumbent's `test_pr_auc` raises `AirflowSkipException`, which skips `reload_serving` too and
  leaves `@champion` alone. `promote` also refuses a version whose artifacts are missing from
  disk — the dangling-alias trap below fails at *serving startup* otherwise.
- **Airflow 3 manual runs of an unscheduled DAG have no logical date**, so `{{ ds }}` and
  `{{ ts_nodash }}` render as undefined and the task fails before docker is contacted. Use
  `{{ dag_run.run_after }}`, which every run has.
- Airflow's metadata tables live in the shared Postgres alongside Label Studio's Django tables.
  `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS=True` means :8090 has no login — the one
  setting to revisit if this ever leaves the laptop.
- The dashboard is plain HTML/SVG in `part2/app/index.html`, no build step and no CDN. Its
  palette comes from the `dataviz` skill's reference instance: categorical slots 1–2 for the
  two latency series, reserved status colors for decisions and breaker state (always paired
  with a text label, never color alone). Dark values are declared under both
  `prefers-color-scheme` and `[data-theme]`.

Part 1's three containers **share nothing but the MLflow registry**.
The decoupling is the pedagogical point: `api-serve/app.py` has never seen `trainer/train.py`.
It locates the model purely through the registry alias, so anything that changes which version
carries `@champion` changes what gets served.

Constraints that are load-bearing and easy to break:

- **`./mlruns` is mounted at the same absolute path `/mlruns` in every container.** Artifacts
  are stored as `file:///mlruns/artifacts/...` URIs with `--no-serve-artifacts`, so a URI
  written by the trainer must resolve identically inside api-serve. Change the mount point in
  one service and model loading breaks with a path error.
- **Promotion is manual.** Notebook section 4 compares runs and sets the `champion` alias.
  Nothing polls or promotes automatically. `api-serve` resolves the alias once at startup;
  `POST /reload` re-resolves it live.
- **The registry is the only copy of a model.** No parallel export to the host, by design.
- **`mlflow.keras.log_model` takes `signature=`, never `input_example=`.** The Keras 3 flavor
  calls `_save_example()` before creating the model dir (still true in mlflow 3.15) and raises
  `FileNotFoundError`. See the comment at `part1/trainer/train.py:99`.
- **Artifacts on disk can outlive-lag the registry.** 29 of the 40 registered `mnist-keras`
  versions have no artifact directory under `mlruns/artifacts/` — the DB rows survived a prune
  that the files did not. A dangling `@champion` fails at *service startup* with
  `MlflowException: No such artifact: ''`, not at promotion time. Before promoting, check the
  version's `storage_location` actually exists on disk.
- Containers run as `${DOCKER_UID}:${DOCKER_GID}` with no `/etc/passwd` entry, so `USER=mlflow`
  and `GIT_PYTHON_REFRESH=quiet` are set to stop `getpass.getuser()` and GitPython from
  failing. Keep them on any new service that talks to MLflow.
- **Backend is `KERAS_BACKEND=torch` everywhere.** Keras 3 is channels_last on all backends,
  including torch — inputs are `(28, 28, 1)`, not `(1, 28, 28)`.

## Data on disk (git-ignored, except the one noted)

```
mlruns/           MLflow backend store (mlflow.db) + artifacts   -> /mlruns everywhere
data/fraud/       Sparkov CSVs, cached mnist.npz                 -> /data
data/derived/     fraud_training.parquet + its .dvc pointer      -> /data
inference/        predictions.jsonl, appended per request        -> /inference
```

`data/derived/*.dvc` is the **only** thing under `data/` that git tracks — 106 bytes naming
84MB by content hash. Everything else there is regenerable from `fetch_data.py` and
`build_dataset.py`.

`api-serve` appends one JSON line per inference to `/inference/predictions.jsonl` via
`_record()`. That file is the observability surface Part 2 is expected to build on (sampling
predictions for human review), so preserve its append-only, one-object-per-line shape.

## Conventions

Both `train.py` and `app.py` are single-file, deliberately short, and comment the *why* rather
than the *what* — comments explain workarounds and design choices, not syntax. Match that.
Configuration arrives through environment variables declared in `docker-compose.yml`; adding a
knob means adding an env var with a default there, not a config file or a CLI flag.
