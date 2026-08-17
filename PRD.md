# PRD: ML Training & Serving Pipeline (Learning Project)

> **Status, 2026-08-17.** This document was the original sketch for Part 2, written before any
> of it was built. It has been corrected against what actually got built and what turned out
> to be wrong. Several original assumptions did not survive contact — they are kept below with
> strikethrough and a note, because *why* they failed is the useful part.
>
> Superseded in places by `README.md` (how to run things) and `CLAUDE.md` (architecture and
> constraints). Where the three disagree, the code wins.

## Overview

End-to-end ML pipeline for learning: train, track, serve, monitor, and collect human feedback
on model predictions.

Part 1 (MNIST, `part1/`) is complete and frozen. Part 2 (fraud detection, `part2/`) is this
document's subject. Both share one MLflow registry.

**Use case: credit-card fraud** on the
[Sparkov synthetic dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data) —
1.3M train / 556k test transactions, 0.58% / 0.39% fraud, split by *time* rather than randomly.

The severe class imbalance is the point, not an obstacle. It forces the model's output to be a
probability under a threshold rather than a label, and a threshold is exactly the contract an
application consumes and a human reviewer disputes.

## Architecture

### Containers

| # | container | status | role |
|---|---|---|---|
| 1 | **mlflow-server** | ✅ built (Part 1) | experiment tracking + model registry |
| 2 | **postgres** | ✅ built | Label Studio store, analyst verdicts, app telemetry |
| 3 | **fraud-trainer** | ✅ built | ephemeral, trains and registers, config via env vars |
| 4 | **fraud-serve** | ✅ built | serves champion from MLflow; `/score` → approve/review/decline, :8001 |
| 5 | **label-studio** | ⬜ next | ~~labelbox~~ self-hosted review queue for sampled inferences |
| 6 | **txn-app** | ⬜ | replays transactions against the endpoint; dashboard on :5500 |
| 7 | **airflow** | ⬜ last | orchestrates train → detect champion → redeploy |

### Data flow

```
fraud-trainer → mlflow registry → @champion alias → fraud-serve
                                                        ↓
                            txn-app scores transactions, applies threshold
                                                        ↓
                               review-band transactions → label-studio
                                                        ↓
                            analyst verdicts → postgres → DVC snapshot
                                                        ↓
                                              next training run
```

The `txn-app` is both the load generator and the lesson: it is the thing that has to survive a
model changing underneath it.

## Corrections to the original plan

**~~Labelbox — open source, self-hosted~~ → Label Studio CE.**
Labelbox is SaaS-only. There is a `labelbox-python` SDK and a `labelbox/node` image on Docker
Hub, but no self-hostable labeling server. Label Studio Community Edition (Apache-2.0,
`heartexlabs/label-studio`, Postgres-backed) is what the original description actually
described, and keeps the "runs on one laptop, no external accounts" property.

**~~BentoML for serving~~ → FastAPI, deferred.**
Part 1's `api-serve` already demonstrates registry-driven serving in ~90 lines. Swapping
frameworks teaches packaging, not MLOps. Revisit once the loop is closed.

**~~Airflow from the start~~ → Airflow last.**
Each DAG task will wrap a command already proven by hand. Building the orchestrator first means
debugging the scheduler instead of the pipeline.

**~~Champion = best accuracy~~ → champion = best PR-AUC.**
At a 0.4% base rate, "never fraud" scores 99.6% accuracy. Accuracy is still logged, purely so
the failure mode is visible in the MLflow UI.

**~~1% random sampling to the review queue~~ → sample the review band.**
Random 1% of a 0.4%-positive stream is almost entirely obvious legitimate transactions —
analyst time spent on nothing. Sampling the *uncertain* band around the threshold is where a
human verdict actually carries information.

**~~test-client with SQLite~~ → txn-app on the shared Postgres.**
The verdicts, the scored transactions and the app's telemetry all want to be queried together.
A separate SQLite would need joining across two stores on the first interesting question.

## Decisions made while building

**Feature engineering is shared code, not duplicated code.** `part2/common/features.py` is
imported by both trainer and server. This breaks Part 1's "the server has never seen the
trainer" property deliberately: date parsing and haversine distance can't be expressed as Keras
layers, and reimplementing them server-side is the classic training/serving skew bug. Learned
state (normalisation constants, embeddings) still lives inside the model artifact.

**Categoricals use the hashing trick, not `StringLookup`.** Keras 3's `StringLookup` requires
TensorFlow even on the torch backend. crc32 into fixed buckets avoids a second framework *and*
leaves no vocabulary to persist alongside the model. crc32 rather than `hash()` because the
latter is salted per process, which would silently desync trainer and server.

**No history-dependent features.** "This card's average spend" would be a strong signal and
would require a feature store at serving time, because the endpoint sees one transaction with
no past. The boundary is left visible rather than papered over.

**The threshold is a model property, not a server setting.** `train.py` selects it on the
validation tail and writes it as an `operating_threshold` run tag. The server reads it from the
model version rather than carrying its own default, so promoting a model promotes its decision
point with it.

## Known issues

- **Dangling registry aliases.** 29 of 40 `mnist-keras` versions have registry rows but no
  artifact files under `mlruns/artifacts/`. A `@champion` alias pointing at one of them fails at
  *service startup*, not at promotion. Cause of the prune is unknown; worth understanding before
  Part 2 writes more artifacts to the same store.
- **Threshold drift.** The operating point holds 0.90 precision on validation and 0.859 on
  test. The test period is later in time with a lower fraud rate. This is the drift the
  retraining loop is meant to answer.

## Tech stack

Keras 3 (torch backend) · MLflow (tracking + registry) · FastAPI + uvicorn (serving) ·
Label Studio CE (human-in-the-loop) · Postgres · DVC (derived datasets only) ·
Airflow (orchestration, last) · Docker Compose

## Out of scope

- Authentication, TLS, secrets management, network isolation — `.env` files are fine here
- Data version control for the *raw* Kaggle CSVs; only the derived dataset is versioned
- GPU training — everything is CPU-sized on purpose

## Future considerations

- Nginx reverse proxy + basic auth if this is ever shared beyond one laptop
- API keys / rate limiting on the serving endpoint
- A vault for secrets when deploying to a company server
- BentoML, revisited as a packaging exercise once the loop is closed
