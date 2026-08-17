"""A transaction application that depends on the fraud endpoint.

This is the part of Part 2 that is not about machine learning at all. It replays
fraudTest.csv as a live stream and has to answer the questions any real consumer of a
model faces:

  - the endpoint is slow -- how long do I wait before giving up?
  - the endpoint is down -- do I approve or decline what I cannot score?
  - the endpoint just swapped models underneath me -- do I notice?

It deliberately has NO ML dependency. No keras, no torch, no features.py -- it forwards raw
transaction rows and consumes decisions. If this app needed to know how the model works, the
serving boundary would be wrong.
"""
import asyncio
import csv
import os
import statistics
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import psycopg
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

SCORE_URL = os.getenv("SCORE_URL", "http://fraud-serve:8001")
DATA_FILE = Path(os.getenv("FRAUD_TEST_CSV", "/data/fraud/fraudTest.csv"))
SCHEMA = Path(os.getenv("SCHEMA_SQL", "/app/schema.sql"))
PG = os.environ["POSTGRES_DSN"]
INDEX = Path(__file__).parent / "index.html"

TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT", "2.0"))
RETRIES = int(os.getenv("RETRIES", "2"))            # total attempts per transaction
FALLBACK = os.getenv("FALLBACK_DECISION", "review")  # what to do when scoring is impossible
CB_THRESHOLD = int(os.getenv("BREAKER_THRESHOLD", "5"))
CB_COOLDOWN = float(os.getenv("BREAKER_COOLDOWN", "10"))

# Columns the endpoint needs, and how to coerce them out of CSV strings.
FLOATS = ["amt", "lat", "long", "merch_lat", "merch_long"]
INTS = ["city_pop"]
STRINGS = ["trans_date_trans_time", "category", "merchant", "gender", "state", "dob", "trans_num"]

STATE = {
    "running": False,
    "rate": float(os.getenv("RATE", "5")),   # transactions per second
    "fault_injection": False,                # send to a black hole, to watch the app cope
    "breaker": "closed",                     # closed | open | half_open
    "consecutive_failures": 0,
    "opened_at": 0.0,
    "sent": 0,
}
# Recent per-request records, for live percentiles and the chart. ~10 minutes at 20/s.
RECENT: deque = deque(maxlen=12000)
PENDING: list = []      # buffered DB writes, flushed once a second


def _connect():
    return psycopg.connect(PG, autocommit=True)


def rows():
    """Stream the test file forever, restarting at EOF. Never loads 150MB into memory."""
    while True:
        with DATA_FILE.open() as f:
            for r in csv.DictReader(f):
                txn = {k: r[k] for k in STRINGS if k in r}
                txn |= {k: float(r[k]) for k in FLOATS}
                txn |= {k: int(float(r[k])) for k in INTS}
                txn["label"] = int(r["is_fraud"])
                yield txn


def breaker_allows() -> bool:
    """Open circuit skips the call entirely -- the point is to stop hammering a dead service.

    After the cooldown it lets exactly one probe through (half-open); success closes it,
    failure re-opens it for another cooldown.
    """
    if STATE["breaker"] == "closed":
        return True
    if STATE["breaker"] == "open":
        if time.monotonic() - STATE["opened_at"] >= CB_COOLDOWN:
            STATE["breaker"] = "half_open"
            return True
        return False
    return True     # half_open: this is the probe


def record_success():
    STATE["consecutive_failures"] = 0
    STATE["breaker"] = "closed"


def record_failure():
    STATE["consecutive_failures"] += 1
    if STATE["breaker"] == "half_open" or STATE["consecutive_failures"] >= CB_THRESHOLD:
        STATE["breaker"] = "open"
        STATE["opened_at"] = time.monotonic()


async def score(client: httpx.AsyncClient, txn: dict) -> dict:
    """One transaction, with retries, a timeout budget, and a fallback that always answers.

    The application must emit a decision for every transaction. 'The model was unavailable'
    is not an answer a checkout flow can return to a customer, so an un-scored transaction
    still gets FALLBACK -- and that choice is a business decision, not a technical one.
    Approve and lose money to fraud, or decline and reject good customers.
    """
    base = "http://127.0.0.1:9" if STATE["fault_injection"] else SCORE_URL
    t0 = time.perf_counter()

    if not breaker_allows():
        return {"outcome": "shed", "decision": FALLBACK, "client_ms": 0.0,
                "server_ms": None, "version": None, "error": "circuit open"}

    err = None
    for attempt in range(RETRIES):
        try:
            r = await client.post(f"{base}/score", json=txn, timeout=TIMEOUT_S)
            r.raise_for_status()
            d = r.json()
            record_success()
            return {"outcome": "scored", "decision": d["decision"],
                    "client_ms": (time.perf_counter() - t0) * 1000,
                    "server_ms": d.get("latency_ms"), "version": d.get("version"),
                    "probability": d.get("probability"), "error": None}
        except Exception as e:                       # noqa: BLE001 -- any failure is a failure
            err = f"{type(e).__name__}: {e}"[:200]
            if attempt + 1 < RETRIES:
                await asyncio.sleep(0.1 * (attempt + 1))    # brief linear backoff

    record_failure()
    return {"outcome": "fallback", "decision": FALLBACK,
            "client_ms": (time.perf_counter() - t0) * 1000,
            "server_ms": None, "version": None, "error": err}


async def replay():
    """The stream. Paced to STATE['rate'], which can change while it runs."""
    stream = rows()
    async with httpx.AsyncClient() as client:
        while True:
            if not STATE["running"]:
                await asyncio.sleep(0.2)
                continue
            started = time.monotonic()
            txn = next(stream)
            res = await score(client, txn)
            res |= {"ts": time.time(), "label": txn["label"], "trans_num": txn.get("trans_num")}
            RECENT.append(res)
            PENDING.append(res)
            STATE["sent"] += 1

            # Pace the loop, accounting for how long the call actually took.
            await asyncio.sleep(max(0.0, 1.0 / max(STATE["rate"], 0.1) - (time.monotonic() - started)))


async def flush_loop():
    """Batch the telemetry writes. One INSERT per transaction would make the app's own
    database latency the bottleneck we are trying to measure in the endpoint."""
    while True:
        await asyncio.sleep(1.0)
        if not PENDING:
            continue
        batch, PENDING[:] = PENDING[:], []
        try:
            with _connect() as c:
                c.cursor().executemany(
                    """INSERT INTO app_requests
                         (trans_num, outcome, decision, client_ms, server_ms,
                          model_version, breaker_state, error)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    [(b["trans_num"], b["outcome"], b["decision"], b["client_ms"],
                      b["server_ms"], b["version"], STATE["breaker"], b["error"])
                     for b in batch])
        except Exception as e:                        # noqa: BLE001
            print(f"[app] telemetry flush failed: {e}", flush=True)


@asynccontextmanager
async def lifespan(app):
    with _connect() as c:
        c.execute(SCHEMA.read_text())
    tasks = [asyncio.create_task(replay()), asyncio.create_task(flush_loop())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Transaction app", lifespan=lifespan)


class Control(BaseModel):
    running: bool | None = None
    rate: float | None = None
    fault_injection: bool | None = None


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(int(q * len(xs)), len(xs) - 1)], 1)


@app.get("/")
def index():
    return FileResponse(INDEX)


@app.get("/api/stats")
def stats():
    now = time.time()
    window = [r for r in RECENT if now - r["ts"] <= 60]
    scored = [r for r in window if r["outcome"] == "scored"]

    # Per-second buckets for the latency chart, last 120s.
    buckets: dict[int, list] = {}
    for r in RECENT:
        if now - r["ts"] <= 120:
            buckets.setdefault(int(r["ts"]), []).append(r)
    series = [{"t": t,
               "client_p95": pct([x["client_ms"] for x in b if x["outcome"] == "scored"], 0.95),
               "server_avg": round(statistics.fmean(s), 1) if (s := [x["server_ms"] for x in b
                                                                    if x["server_ms"] is not None]) else None,
               "n": len(b)}
              for t, b in sorted(buckets.items())]

    return {
        "state": {k: STATE[k] for k in
                  ("running", "rate", "fault_injection", "breaker", "consecutive_failures", "sent")},
        "window_seconds": 60,
        # Divide by the span actually covered, not a flat 60 -- otherwise the first minute
        # after startup reports a fraction of the true rate and looks like a throughput bug.
        "throughput": round(len(window) / max(now - window[0]["ts"], 1.0), 2) if window else 0.0,
        "decisions": Counter(r["decision"] for r in window),
        "outcomes": Counter(r["outcome"] for r in window),
        "latency": {"client_p50": pct([r["client_ms"] for r in scored], 0.50),
                    "client_p95": pct([r["client_ms"] for r in scored], 0.95),
                    "client_p99": pct([r["client_ms"] for r in scored], 0.99),
                    "server_avg": round(statistics.fmean(s), 1) if (s := [r["server_ms"] for r in scored
                                                                          if r["server_ms"] is not None]) else None},
        "versions": Counter(r["version"] for r in window if r["version"]),
        "fallback_rate": round(sum(1 for r in window if r["outcome"] != "scored") / len(window), 4) if window else 0.0,
        "series": series,
    }


@app.post("/api/control")
def control(c: Control):
    for k, v in c.model_dump(exclude_none=True).items():
        STATE[k] = v
    if c.fault_injection is False:      # clearing the fault should not leave the breaker stuck
        STATE["breaker"], STATE["consecutive_failures"] = "closed", 0
    return STATE


@app.post("/api/reload")
async def reload_upstream():
    """Ask the endpoint to re-resolve @champion, without pausing the stream.

    Watch the version tile and the latency chart while this runs: loading a model takes
    seconds, and the app keeps having to answer throughout.
    """
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{SCORE_URL}/reload", timeout=120)
        return r.json()
