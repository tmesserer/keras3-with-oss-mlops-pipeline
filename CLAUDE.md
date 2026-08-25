# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning project for MLOps, not a production system.

**Part 1** (`part1/`, complete — code frozen, notebook still maintained) is a
**train → compare → promote → serve → observe**
loop on MNIST, using Keras 3 on the torch backend with MLflow as tracking server and registry.

**Part 2** (`part2/`, complete through orchestration) moves to fraud detection on the Sparkov
credit-card dataset and adds what Part 1 can't teach: an application depending on the endpoint,
a human review queue feeding labels back, and a versioned derived dataset. Both parts share one
MLflow registry and one Postgres.

**Students start at `slides/`, then run the notebooks.** The decks carry the concepts (see
"The slides" at the end of this file); the notebooks carry the operations.

**The two notebooks are the documentation.** `part1/notebooks/hub.ipynb` and
`part2/notebooks/hub.ipynb` carry every explanation this project has; `README.md` is one
paragraph on purpose and must stay that way, and there is no PRD (it was folded into the Part 2
notebook — `git show HEAD:PRD.md` if you want the original sketch, which was wrong about
Labelbox and about accuracy-based selection). Explanation belongs in a markdown cell, not in a
new markdown file.

**Each part's notebook is organised as lettered stages, and each part has exactly one diagram**
— `part1/notebooks/architecture.svg` and `part2/notebooks/architecture.svg`, referenced once at
the top of their notebook and never duplicated per section. Both are drawn in the same visual
language and palette, and Part 1's ends with a "NOT HERE YET" band naming what Part 2 adds. The
stage codes are the shared vocabulary between notebook, diagram and source.

Part 1 — A data · B train · C select · D serve · E observe, around one loop the student closes
by hand (which is the setup for Part 2's three automated ones):

| code | notebook section | code lives in |
|---|---|---|
| A1 / A2 | cached `mnist.npz` · look at the digits | notebook only |
| B1 / B2 / B3 | build images · mlflow-server · three runs | `part1/trainer/train.py` (B3) |
| C1 / C2 | compare on accuracy · move `@champion` | notebook only |
| D1 / D2 | `api-serve` · `/predict` | `part1/api-serve/app.py` |
| E1 | `predictions.jsonl` | `api-serve/app.py` `_record()` |

Part 2 — seven stages and three explicit loops (① decision, ② label, ③ improvement):

| code | notebook section | code lives in |
|---|---|---|
| A1 / A2 / A3 | fetch · explore · features | `pipeline/fetch_data.py` · notebook only · `common/features.py` |
| B1 / B2 / B3 | train · threshold · promote | `trainer/train.py` (B1–B2); B3 is the notebook and `airflow/dags/fraud_retrain.py` |
| C1 / C2 | endpoint · decision band | `serving/app.py` |
| D1 / D2 | consumer · failure handling | `app/app.py` |
| E1 / E2 | review queue · verdicts back | `labeling/label_config.xml` · `labeling/sync.py` |
| F1 / F2 | build snapshot · hash on the run | `pipeline/build_dataset.py` · `trainer/train.py` |
| G1 / G2 | retrain DAG · label-sync DAG | `airflow/dags/` |

Module docstrings name their code. Adding or moving a stage means updating the SVG, the
notebook and the docstring together, or the vocabulary rots.

## Commands

Everything runs through Docker Compose. There is no test suite, no linter, and no build system
beyond the images themselves.

`.env` sets `COMPOSE_FILE=docker-compose.yml:compose.part1.yml`, which is what lets a bare
`docker compose` see every service without `-f` flags. **If `.env` is missing, only the base
file loads and `trainer`/`api-serve` appear not to exist.** Both notebooks' first cell writes
it from `.env.example`, substituting `DOCKER_UID`/`DOCKER_GID`/`DOCKER_SOCK_GID` for this
machine's, and never overwrites an existing `.env` (it may hold edited Label Studio
credentials). By hand it is `cp .env.example .env` plus the `sed` line in that file's header.

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

The host-side workflow is a notebook per part — `part1/notebooks/hub.ipynb` and
`part2/notebooks/hub.ipynb` — driving all of the above via `!docker compose` and the MLflow
client. Both first cells walk up to the repo root (the folder holding `docker-compose.yml`) and
`chdir` there, so every `!docker compose` line resolves regardless of where the notebook was
launched from. Part 2 uses `--profile part2` for everything, never `serve2`/`label`/`orchestrate`
alone: `postgres` is not in those profiles, and compose refuses to start a service whose
`depends_on` target is inactive. The host venv holds clients only — MLflow, pandas, numpy,
requests, matplotlib, dvc, psycopg, jupyterlab.
Keras and torch live exclusively inside the containers; the one exception is
`part2/common/features.py`, which the notebook imports directly in A3 because it depends on
nothing but pandas and numpy.

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
  `prefers-color-scheme` and `[data-theme]`. Both notebooks' A2 charts and both
  `architecture.svg` files use the same slots — one palette across the project.

Part 1's three containers **share nothing but the MLflow registry**.
The decoupling is the pedagogical point: `api-serve/app.py` has never seen `trainer/train.py`.
It locates the model purely through the registry alias, so anything that changes which version
carries `@champion` changes what gets served.

Constraints that are load-bearing and easy to break:

- **`./mlruns` is mounted at the same absolute path `/mlruns` in every container.** Artifacts
  are stored as `file:///mlruns/artifacts/...` URIs with `--no-serve-artifacts`, so a URI
  written by the trainer must resolve identically inside api-serve. Change the mount point in
  one service and model loading breaks with a path error.
- **`mlruns/.gitkeep` is tracked, and `.gitignore` says `mlruns/*` so it survives.** A bind
  mount whose host directory does not exist is created by the docker daemon as **root**, and
  `mlflow-server` runs as `${DOCKER_UID}` — it then cannot create `mlflow.db` and dies with
  `sqlite3.OperationalError: unable to open database file`. Shipping the directory in git is
  what makes a fresh clone work. Same reason `inference/.gitkeep` exists; `data/` is covered by
  the tracked `data/derived/*.dvc`. If it ever does go root-owned:
  `docker run --rm -u 0 -v "$PWD/mlruns:/mlruns" alpine:3 chown -R $(id -u):$(id -g) /mlruns`.
- **Promotion is manual.** Notebook section C1–C2 compares runs and sets the `champion` alias.
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

## The slides

**`slides/` is where a student starts.** Three decks, opened straight from the filesystem —
`Overview_slides.html` (course introduction, 2 slides), `part1.html` (10 slides) and `part2.html`
(29 slides). The audience has no IT background, so the decks
define the vocabulary the notebooks assume: container, image, layer, registry, volume,
endpoint. They do not define ML basics.

- **Slides own the concepts, the notebooks own the operations.** This narrows the older rule
  above: explanation still never goes into a new markdown *file*, but a concept explained on a
  slide must not be explained again in a notebook cell. The notebook says what a cell does and
  how long it takes; the slide says why the thing exists. Both notebooks were trimmed to match
  on 2026-08-20 and now point at slide numbers ("slide 17 explains why…"). What a trimmed
  notebook keeps beyond the operations is the repo-specific *why* no slide carries: the crc32
  salting, the `label-sync.local` alias, the `data/*` gitignore rule, the MLflow 3
  `models:/m-<id>` resolution, the known rough edges. Renumbering a deck means re-checking those
  pointers.
- **The only sanctioned copies of a diagram live in the decks**, which inline the
  `architecture.svg` files so their stages can be revealed one press at a time. Each copy drops
  the file's own `<style>` and takes its colours from `deck.css` instead, and its class names
  are prefixed `a-` so nothing collides. Both source SVGs wrap each stage in `<g id="dg-…">` /
  `<g id="fl-…">`; those ids are the reveal order and are load-bearing for the decks, harmless
  in the notebooks.
- **`part2.html` inlines that diagram three times, once per loop** — stages A–D and Postgres,
  then E, then F and the G band — with the earlier stages kept visible under `.a-dim`. The
  three copies carry `m1-`/`m2-`/`m3-` marker id prefixes so they can share one document. They
  were produced mechanically from the source SVG — drop `<style>`, prefix classes with `a-`,
  prefix marker ids, replace the hard-coded loop colours with `var(--c1..3)`, then mark each
  stage group `frag`/`a-dim` or delete it. If the source SVG changes, redo that transformation
  for all three rather than patching one copy by hand.
- **`part2.html` declares `--good`/`--warning`/`--critical` locally**, copied from
  `part2/app/index.html`, so one decision wears one colour across the whole project. They are
  never used for a data series and never appear without a word beside them.
- **`deck.css` and `deck.js` are shared by all three decks.** No build step, no CDN, no
  modules — a classic `<script src>` so `file://` behaves like `http://`. Everything is sized
  from `--u`, one unit of a fixed 16:9 canvas, so a slide looks the same on a projector and
  prints to PDF unchanged.
- **Navigation is two independent axes.** Up/Down moves between slides, Left/Right steps the
  reveals inside one slide. Right on the last reveal does nothing on purpose: the axes never
  mix, so you cannot lose your place while talking. `f` is full screen, `d` toggles the theme.
  Reveal order is `data-frag="N"`, numeric, and elements sharing an N appear together.
- **Every slide carries the same footnote**, written once by `deck.js`: *Slides and project
  were created using AI, really.*
- The palette is the project palette — the same `dataviz` slots used by both
  `architecture.svg` files, both notebooks' A2 charts and the Part 2 dashboard.

Part 1's ten slides: the map (the diagram, revealed in eight steps) · training vs inference ·
two evaluations · what MLflow is · why containers · image layers · `CMD` and env at launch ·
volumes · the dataset · serving and Pydantic.

Part 2's twenty-nine, in six acts: **the step up** (what Part 2 adds · docker compose ·
Kubernetes) · **data and model** (map ① and Postgres · four facts about the data · PR-AUC and
the threshold · what `features.py` computes · the two rules it obeys) · **loop ①** (the decision band · the simulated
application · timeout, fallback, breaker · what an outage costs) · **loop ②** (map ② · what you
cannot know without a human · labels are bought · build the review tool yourself · label-sync ·
HITL is not free) · **loop ③** (map ③ · what a hash is · parquet and a stable write · a dataset
with an identity · Airflow by analogy · the two DAGs) · **out of the classroom** (the adoption
ladder · one run written down · six questions an auditor asks · where an LLM plugs in · the
closing map in words). Two charts on
the outage slide use assumed costs, stated in the caption; the latencies are the real ones
from D2.

