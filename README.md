# Keras 3 + MLflow — an MLOps learning project

Two parts, sharing one MLflow registry and one Postgres.

- **Part 1** (`part1/`) — MNIST. A complete **train → compare → promote → serve → observe**
  loop. Done, and deliberately frozen.
- **Part 2** (`part2/`) — credit-card fraud. Adds the things Part 1 can't teach: a real
  application depending on the endpoint, a human review queue feeding labels back, and a
  versioned derived dataset. In progress — trainer, serving and the consumer app are
  built; the review queue is next.

## Setup (WSL or Linux)

```bash
cp .env.example .env
sed -i "s/^DOCKER_UID=.*/DOCKER_UID=$(id -u)/;s/^DOCKER_GID=.*/DOCKER_GID=$(id -g)/" .env

python3.12 -m venv .venv && source .venv/bin/activate   # 3.12, not 3.14: pyarrow has no cp314 wheels
pip install -r requirements-host.txt
```

`.env` also sets `COMPOSE_FILE`, merging `docker-compose.yml` (shared infrastructure) with
`compose.part1.yml` and `compose.part2.yml`. That is why a bare `docker compose` finds every
service without `-f` flags — **without `.env`, only the base file loads and the trainers appear
not to exist.**

The host venv holds clients only — the MLflow client, pandas, gdown. Keras and torch live
exclusively inside the containers.

### Layout

```
docker-compose.yml     shared: mlflow-server, postgres
compose.part1.yml      trainer, api-serve
compose.part2.yml      fraud-trainer, fraud-serve, txn-app, label-studio, label-sync
part1/                 trainer/, api-serve/, notebooks/hub.ipynb      [frozen]
part2/                 trainer/, serving/, app/, labeling/, common/, pipeline/
mlruns/                MLflow backend store + artifacts  -> /mlruns in every container
data/                  mnist.npz, fraud/*.csv            -> /data
inference/             predictions.jsonl, per request    -> /inference
```

`mlruns/`, `data/` and `inference/` are git-ignored. `./mlruns` is mounted at the *same absolute
path* in every container, so `file:///mlruns/artifacts` URIs resolve identically from any of
them — that constraint is load-bearing, don't remount it elsewhere.

## Part 1 — MNIST

Three containers that share nothing but the MLflow registry.

| container | role | lifecycle |
|---|---|---|
| `mlflow-server` | tracking + model registry (sqlite + file artifacts) | long running, :5000 |
| `trainer` | Keras 3 on the torch backend, one run per launch | launched per run, exits |
| `api-serve` | FastAPI + uvicorn serving `models:/mnist-keras@champion` | you start it, :8000 |

The split is the point: the trainer is ephemeral and configured entirely from outside, the
server outlives it, and `api-serve` has never seen `train.py` — it finds the model through the
registry alias alone.

```bash
jupyter lab part1/notebooks/hub.ipynb   # or open the folder in VS Code and pick .venv as kernel
```

Work through the notebook top to bottom. Section 3 trains three models, one cell each; change a
number in any cell and re-run it — nothing is rebuilt, because the hyperparameters live outside
the image. The first cell walks up to the repo root, so the notebook works wherever it's opened
from.

Manual equivalents:

```bash
docker compose up -d mlflow-server                       # http://localhost:5000

docker compose --profile tools run --rm \
  -e RUN_NAME=cnn-32 -e KIND=cnn -e SIZE=32 -e LR=1e-3 trainer

docker compose --profile serve up -d api-serve           # http://localhost:8000/health
curl localhost:8000/predict/sample/7
tail -f inference/predictions.jsonl
docker compose --profile tools --profile serve down
```

### Notes

- **Champion selection**: notebook step 5 sets the `champion` alias on the best registered
  version. `api-serve` resolves the alias at startup; `POST /reload` re-resolves it live.
  Promote only to a version whose artifacts still exist on disk — a dangling alias fails at
  *service startup* with `No such artifact: ''`, not at promotion time. 29 of the 40
  `mnist-keras` versions currently have registry rows but no files, so this is not theoretical.
- **Where models live**: only in the MLflow registry, as the `keras` flavor. No second copy on
  the host — the registry is the source of truth.
- **Parameters**: the trainer reads its whole config from env vars (`RUN_NAME`, `KIND`, `SIZE`,
  `DROPOUT`, `OPT`, `LR`, `EPOCHS`, `BATCH_SIZE`, `TRAIN_SUBSET`). `compose.part1.yml` holds the
  defaults; each launch overrides them with `-e`. The image never needs rebuilding.
- **Speed**: defaults are `EPOCHS=3`, `TRAIN_SUBSET=20000` (CPU friendly). `TRAIN_SUBSET=0` for
  the full 60k.
- **Backend**: `KERAS_BACKEND=torch` everywhere. Swap to `tensorflow` or `jax` and the same code
  trains and serves unchanged.

## Part 2 — fraud detection

The [Sparkov synthetic credit-card set](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data):
1.3M training transactions, 556k test, **0.58% / 0.39% fraud**. `fraudTrain.csv` and
`fraudTest.csv` are a *time* split — train ends 2020-06-21, test begins 78 seconds later.

```bash
python part2/pipeline/fetch_data.py    # ~500 MB into data/fraud/, idempotent

docker compose --profile part2 run --rm \
  -e RUN_NAME=mlp-128-64-full -e EPOCHS=20 fraud-trainer
```

Roughly 4 minutes on CPU for the full 20 epochs. `TRAIN_SUBSET=200000 EPOCHS=2` gives a
30-second smoke run.

### Why this dataset changes the lesson

At a 0.4% base rate, **accuracy is a lie**: predicting "never fraud" scores 99.6%. The trainer
logs accuracy anyway, precisely so you can watch it stay near-perfect while the model misses a
third of all fraud. Champion selection uses **PR-AUC** instead.

The current champion (`fraud-keras` v2) scores:

| metric | value |
|---|---|
| test PR-AUC | 0.777 — against a 0.00386 base rate |
| precision / recall at threshold | 0.859 / 0.643 |
| test accuracy | 0.99822 — the meaningless one |

The model outputs a probability, not a verdict, so a **threshold** turns it into a decision.
`train.py` picks that threshold on the validation tail (targeting `TARGET_PRECISION`, default
0.90) and records it as an `operating_threshold` tag on the run, so the serving layer inherits
the decision point instead of inventing one.

Note that the threshold held 0.90 precision on validation but delivered 0.859 on test. The test
period is a later six months with a lower fraud rate: real temporal drift, and the reason the
retraining loop exists.

### Feature design

`part2/common/features.py` turns raw transaction rows into model inputs, and is shared
verbatim by the trainer and the serving image. It parses dates and does arithmetic; it
learns nothing. Normalisation constants and embeddings live inside the Keras model, so the
registry artifact stays self-contained.

Two decisions worth knowing about:

- **Categoricals use the hashing trick** (crc32 → fixed buckets: merchant→2048, state→128,
  category→32, gender→4) rather than `StringLookup`, which requires TensorFlow even on the
  torch backend. crc32 specifically, because Python's `hash()` is salted per process — trainer
  and server would disagree on every bucket. Bonus: no learned vocabulary to persist and keep
  in sync with the model.
- **Every feature is row-local.** No "this card's average spend" — that needs history the
  endpoint doesn't have when scoring a single transaction. Stateful features require a feature
  store, and that boundary is more instructive than a quietly leaky feature.

Sharing `features.py` between trainer and server is a deliberate break from Part 1's "the
server has never seen the trainer" property. The alternative — reimplementing date parsing and
haversine distance in the server — is the classic training/serving skew bug.

### Serving

```bash
docker compose --profile part2 up -d fraud-serve     # :8001, brings postgres up with it
curl localhost:8001/health
curl localhost:8001/score/sample/1767                # a known fraud from fraudTest.csv
curl localhost:8001/stats
```

`/score` takes a raw transaction — the CSV columns, unprocessed — and returns a **decision**,
not just a probability:

| endpoint | does |
|---|---|
| `POST /score` | raw transaction → `{probability, decision, thresholds, latency_ms, id}` |
| `GET /score/sample/{i}` | same, pulling row `i` out of `fraudTest.csv` |
| `GET /health` | model version, both thresholds, and where they came from |
| `POST /reload` | re-resolve `@champion` live, picking up a new threshold with it |
| `GET /stats` | decision mix, mean latency and caught-fraud counts, from Postgres |

Decisions are three-way. Above `threshold_decline` (the model's own operating point) it
declines; above `threshold_review` (half of it by default, via `REVIEW_SPAN`) it asks a human;
otherwise it approves. That middle band is the queue Phase 4 hands to Label Studio — sampling
the *uncertain* transactions rather than a random 1%, since random sampling of a 0.4%-positive
stream is nearly all obvious legitimate traffic.

`fraud-serve` refuses to start against a model with no `operating_threshold` tag rather than
falling back to 0.5 — at this base rate a wrong default silently declines a large fraction of
legitimate traffic. Set `DECLINE_THRESHOLD` / `REVIEW_THRESHOLD` in `compose.part2.yml` to
override the model's choice and watch the decision mix move.

Every score is written to `scored_transactions` in Postgres (schema in
`part2/pipeline/schema.sql`, applied idempotently at startup) and appended to
`inference/fraud_scores.jsonl`. Postgres is the spine: the review queue selects from it, and
the retraining snapshot joins it to the `verdicts` table.

### The consuming application

```bash
docker compose --profile part2 up -d txn-app     # dashboard on http://localhost:5500
```

`txn-app` replays `fraudTest.csv` as a live stream against `/score`. It is the part of Part 2
with **no machine learning in it at all** — no keras, no torch, no `features.py`, a ~100MB
image. It forwards raw rows and consumes decisions. If it ever needs an ML dependency, the
serving boundary has leaked.

Its job is to answer the questions every real consumer of a model faces:

- **The endpoint is slow.** `REQUEST_TIMEOUT` (2s) bounds the wait, `RETRIES` (2) bounds the
  attempts.
- **The endpoint is down.** Every transaction still needs an answer, so an un-scored one gets
  `FALLBACK_DECISION`. That default is `review` — because `approve` loses money to fraud and
  `decline` rejects good customers. There is no free option; it's a business decision wearing
  a config flag.
- **The endpoint is dead, not just slow.** After `BREAKER_THRESHOLD` consecutive failures the
  circuit opens and requests are *shed* — answered from the fallback without a call — until a
  single probe is allowed through every `BREAKER_COOLDOWN` seconds.
- **The model changed underneath it.** Every decision is stamped with the model version, so a
  champion swap shows up as a shift in the decision mix mid-stream.

The **Simulate outage** button points the app at a dead address without touching any
container, so you can watch the breaker trip and recover. Measured during a real outage run:

| outcome | mean client latency |
|---|---|
| scored | 20.9 ms |
| fallback (timeout + retry) | 102.2 ms |
| shed (circuit open) | 0.0 ms |

**Failing is slower than succeeding** — that asymmetry is the whole argument for a circuit
breaker, and it's visible in your own telemetry rather than asserted in a blog post.

`app_requests` in Postgres records what the *client* saw, deliberately separate from
`scored_transactions`, which records what the *server* did. The gap between `client_ms` and
`server_ms` is network, queueing and retry — time the model never sees. Rows where
`outcome != 'scored'` have no server-side counterpart at all: those are the transactions the
app had to decide without a model.

### The human in the loop

```bash
docker compose --profile label up -d          # label-studio :8080, label-sync :8002
```

Log into Label Studio with the `LS_USERNAME` / `LS_PASSWORD` from `.env` and open the **Fraud
review queue** project. It is already full: `label-sync` sweeps the review band every 30
seconds and pushes anything not yet queued, highest probability first — so a half-finished
queue is the half worth finishing.

Annotate a task and the verdict is in Postgres before you reach the next one:

```bash
curl -s localhost:8002/health                 # queue depth, verdict count, last push
docker compose exec postgres psql -U mlops -d mlops \
  -c "select v.verdict, s.probability, s.label from verdicts v
        join scored_transactions s on s.id = v.scored_id"
```

Three things here are deliberate:

- **The analyst never sees the model's score.** `label_config.xml` renders the amount,
  merchant, distance and cardholder — not the probability, not the decision. An analyst shown
  "0.97 fraud" agrees with the model, and a queue that only confirms the model produces labels
  with no information in them. The score decides *what* gets reviewed; it must not decide the
  answer.
- **Two return paths, because webhooks lie.** Label Studio `POST`s each annotation to
  `label-sync`, which is the fast path. `POST localhost:8002/pull` re-reads every annotation
  from the API and upserts — that's the correct one, and the only thing that recovers a
  delivery lost while the container was down. Verdicts key on `scored_id`, so replaying is
  free and an analyst correcting themselves overwrites.
- **`review_tasks` sits between the two tables.** It makes pushing idempotent and lets a
  webhook carrying only a task id find the score behind it.

`label-sync` has no ML dependency either — same rule as `txn-app`. It moves rows and JSON, and
the only thing it knows about the model is the version string it stamps on each verdict, so a
later run can ask *what did v2 get wrong?*

### The versioned snapshot

```bash
python part2/pipeline/build_dataset.py     # host script -- dvc needs the git repo
```

Joins the 1.3M original rows to whatever the analysts have ruled on and writes one parquet,
then `dvc add`s it:

```
[build] 1,296,678 rows  (3 analyst, 1,296,675 original)  fraud=0.579%
[build] data/derived/fraud_training.parquet  84.1 MB
[build] dvc md5 f39848930e4bdeacf470d3c05175278b  -> commit ...parquet.dvc
```

**The pointer is committed; the parquet is not.** `.gitignore` uses `data/*` rather than
`data/` precisely so `!data/derived/*.dvc` can re-include it — git cannot un-include a file
whose parent directory is excluded. Git carries an exact identity for an 84MB file it never
stores, and the trainer stamps that identity on the run:

```
[data] {'dataset': 'derived-snapshot', 'dvc_md5': 'f398489...', 'analyst_rows': '3',
        'test_rows_withheld': '3'}
```

Three properties are worth checking yourself:

- **The hash is content, not time.** Rebuilding unchanged data reproduces it byte for byte;
  deleting three verdicts and rebuilding returns to the *previous* hash exactly. Rows are
  sorted stably by time before writing for this reason — a snapshot whose hash churns on
  rebuild versions nothing.
- **Analyst rows are a biased sample.** Only the review band is ever queued, so every human
  label sits near the operating threshold. That's where labels are worth buying, and it means
  the snapshot's base rate isn't the population's. `analyst_rows` is tagged so the effect stays
  traceable.
- **`test_rows_withheld` is the leakage guard.** Every analyst row *is* a `fraudTest.csv` row
  that a human relabelled, so the moment training data comes from production the original test
  set stops being held out. The trainer drops reviewed transactions from test — otherwise
  test PR-AUC quietly inflates and stops being comparable with earlier runs.

If the snapshot doesn't exist, the trainer falls back to `fraudTrain.csv` and tags the run
`dataset=raw-csv`, so a fresh clone that has never labelled anything still trains.

There is **no DVC remote**: the cache is local, in `.dvc/cache`. Versioning here is about
identity, not backup.

### Still to build

Airflow, wrapping the steps once they're proven by hand: build → train → compare → promote →
`POST /reload`, plus the label push/pull on a schedule.
