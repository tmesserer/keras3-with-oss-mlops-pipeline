"""C1-C2 · Scores one transaction and returns a DECISION, not a probability.

The difference from Part 1's api-serve is the threshold. A fraud model emits P(fraud); turning
that into approve / review / decline needs an operating point, and that point is a property of
the *model*, not of this server -- so it is read from the model version's run tag
(`operating_threshold`, chosen by the trainer on validation data). Env vars can override it,
which is how the application gets to watch its decision mix move under a changing threshold.

Every score lands in Postgres. The review band is what feeds the human queue in section E, and
that only works if the transactions are queryable rather than sitting in a log file.
"""
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.keras
import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

import features

MODEL_URI = os.getenv("MODEL_URI", "models:/fraud-keras@champion")
SCHEMA = Path(os.getenv("SCHEMA_SQL", "/app/schema.sql"))
DATA_FILE = Path(os.getenv("FRAUD_TEST_CSV", "/data/fraud/fraudTest.csv"))
INFER_LOG = Path(os.getenv("INFER_LOG", "/inference/fraud_scores.jsonl"))
PG = os.environ["POSTGRES_DSN"]

# Empty string means "inherit from the model"; a number overrides it.
ENV_DECLINE = os.getenv("DECLINE_THRESHOLD", "").strip()
ENV_REVIEW = os.getenv("REVIEW_THRESHOLD", "").strip()
# How far below the decline point the review band reaches, when not set explicitly.
REVIEW_SPAN = float(os.getenv("REVIEW_SPAN", "0.5"))

STATE: dict = {}


def _connect():
    return psycopg.connect(PG, autocommit=True)


def load_champion():
    """models:/<name>@<alias> -> concrete version, model, and its operating threshold."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    name, alias = MODEL_URI.removeprefix("models:/").split("@")
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)

    # The trainer wrote the threshold as a run tag. Falling back to 0.5 would be a silent
    # disaster at this base rate, so an untagged model is refused outright.
    tags = client.get_run(mv.run_id).data.tags
    tagged = tags.get("operating_threshold")
    if tagged is None and not ENV_DECLINE:
        raise RuntimeError(
            f"{name} v{mv.version} has no operating_threshold tag and DECLINE_THRESHOLD is "
            f"unset. Refusing to guess a threshold for a {tags.get('task', 'fraud')} model."
        )

    decline = float(ENV_DECLINE) if ENV_DECLINE else float(tagged)
    # Default review band: everything from halfway-to-decline up to the decline point.
    review = float(ENV_REVIEW) if ENV_REVIEW else decline * REVIEW_SPAN

    STATE.update(model=mlflow.keras.load_model(MODEL_URI), model_name=name, alias=alias,
                 version=mv.version, run_id=mv.run_id,
                 threshold_decline=decline, threshold_review=review,
                 threshold_source="env" if ENV_DECLINE else "model tag",
                 loaded_at=datetime.now(timezone.utc).isoformat())
    print(f"[serve] loaded {name} v{mv.version} (@{alias})  "
          f"review>={review:.6f} decline>={decline:.6f} ({STATE['threshold_source']})", flush=True)


@asynccontextmanager
async def lifespan(app):
    with _connect() as c:
        c.execute(SCHEMA.read_text())
    load_champion()
    yield


app = FastAPI(title="Fraud scoring", lifespan=lifespan)


class Transaction(BaseModel):
    """Raw transaction, exactly as it appears in the CSV. All feature work happens here."""
    trans_date_trans_time: str
    amt: float
    category: str
    merchant: str
    gender: str
    state: str
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city_pop: int
    dob: str
    trans_num: str | None = None
    label: int | None = Field(default=None, description="ground truth, if the caller knows it")


def _decide(p: float) -> str:
    if p >= STATE["threshold_decline"]:
        return "decline"
    return "review" if p >= STATE["threshold_review"] else "approve"


def score_one(txn: dict) -> dict:
    t0 = time.perf_counter()
    row = pd.DataFrame([txn])
    p = float(STATE["model"].predict(features.build(row), verbose=0).ravel()[0])
    latency_ms = (time.perf_counter() - t0) * 1000

    result = {"probability": round(p, 6), "decision": _decide(p),
              "model": STATE["model_name"], "version": STATE["version"],
              "threshold_review": STATE["threshold_review"],
              "threshold_decline": STATE["threshold_decline"],
              "latency_ms": round(latency_ms, 2)}

    label = txn.get("label")
    with _connect() as c:
        # ON CONFLICT: replaying the same test file twice must not fail. The score is
        # updated in place, so the row always reflects the most recent model to see it.
        r = c.execute(
            """INSERT INTO scored_transactions
                 (trans_num, model_name, model_version, probability, decision,
                  threshold_review, threshold_decline, latency_ms, payload, label)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (trans_num) DO UPDATE SET
                 scored_at=now(), model_version=EXCLUDED.model_version,
                 probability=EXCLUDED.probability, decision=EXCLUDED.decision,
                 threshold_review=EXCLUDED.threshold_review,
                 threshold_decline=EXCLUDED.threshold_decline,
                 latency_ms=EXCLUDED.latency_ms
               RETURNING id""",
            (txn.get("trans_num"), STATE["model_name"], STATE["version"], p, result["decision"],
             STATE["threshold_review"], STATE["threshold_decline"], latency_ms,
             Jsonb(txn), label)).fetchone()
    result["id"] = r[0]

    INFER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with INFER_LOG.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                            "label": label, **result}) + "\n")
    return result


@app.get("/health")
def health():
    return {k: v for k, v in STATE.items() if k != "model"}


@app.post("/reload")
def reload_model():
    """Re-resolve @champion without a restart. The app should survive this mid-stream."""
    load_champion()
    return health()


@app.post("/score")
def score(txn: Transaction):
    return score_one(txn.model_dump())


@app.get("/score/sample/{i}")
def score_sample(i: int):
    """Simplest possible demo: take row i out of fraudTest.csv and score it."""
    if not DATA_FILE.exists():
        raise HTTPException(404, f"{DATA_FILE} missing -- run part2/pipeline/fetch_data.py")
    try:
        row = pd.read_csv(DATA_FILE, skiprows=range(1, i + 1), nrows=1,
                          usecols=features.RAW + ["trans_num", "is_fraud"])
    except ValueError:
        raise HTTPException(400, "index out of range")
    if row.empty:
        raise HTTPException(400, "index out of range")
    txn = row.iloc[0].to_dict()
    txn["label"] = int(txn.pop("is_fraud"))
    return score_one(txn)


@app.get("/stats")
def stats():
    """Decision mix and latency, straight out of Postgres. The dashboard builds on this."""
    with _connect() as c:
        rows = c.execute(
            """SELECT model_version, decision, count(*), avg(latency_ms)::real,
                      avg(probability)::real
               FROM scored_transactions GROUP BY 1,2 ORDER BY 1,2""").fetchall()
        caught = c.execute(
            """SELECT count(*) FILTER (WHERE label=1 AND decision='decline'),
                      count(*) FILTER (WHERE label=1),
                      count(*) FILTER (WHERE label=0 AND decision='decline')
               FROM scored_transactions WHERE label IS NOT NULL""").fetchone()
    return {
        "by_version": [{"version": v, "decision": d, "count": n,
                        "avg_latency_ms": round(lat, 2), "avg_probability": round(p, 6)}
                       for v, d, n, lat, p in rows],
        "labelled": {"fraud_declined": caught[0], "fraud_total": caught[1],
                     "false_declines": caught[2]},
    }
