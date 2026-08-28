"""E1-E2 · The human in the loop: review band out to Label Studio, analyst verdicts back.

Two directions, and they fail differently, which is why they are separate code paths:

  push  - a loop that hands unreviewed review-band transactions to Label Studio as tasks.
          Idempotent through `review_tasks`; safe to run every minute forever.
  back  - Label Studio calls POST /webhook the moment an annotation is saved. A webhook is
          sent once and never confirmed, so POST /pull re-reads every annotation from the
          API and upserts it. That recovers anything missed while this container was down.

Like txn-app, this service has NO ML dependency. It moves rows and JSON; it never asks what
the model thought. The only thing it knows about the model is the version string it stamps
on a verdict, so a later run can ask "what did v2 get wrong?".
"""
import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg
from fastapi import FastAPI, Request

LS_URL = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080").rstrip("/")
LS_TOKEN = os.environ["LABEL_STUDIO_TOKEN"]
PROJECT_TITLE = os.getenv("LS_PROJECT_TITLE", "Fraud review queue")
CONFIG = Path(os.getenv("LABEL_CONFIG", "/app/label_config.xml"))
SCHEMA = Path(os.getenv("SCHEMA_SQL", "/app/schema.sql"))
PG = os.environ["POSTGRES_DSN"]

# Where Label Studio should call this service back. Must be reachable from inside ITS
# container, so it is a compose service name -- localhost would point Label Studio at itself.
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "http://label-sync:8002/webhook")

PUSH_INTERVAL = float(os.getenv("PUSH_INTERVAL", "30"))
PUSH_BATCH = int(os.getenv("PUSH_BATCH", "50"))

HEADERS = {"Authorization": f"Token {LS_TOKEN}"}
STATE: dict = {"project_id": None, "pushed": 0, "verdicts": 0, "last_push": None, "last_error": None}

# The Choice values in label_config.xml, mapped to what the verdicts table stores.
VERDICT_VALUES = {"fraud": 1, "legitimate": 0}


def _connect():
    return psycopg.connect(PG, autocommit=True)


def _client():
    return httpx.Client(base_url=LS_URL, headers=HEADERS, timeout=30)


# ------------------------------------------------------------------ setup

def ensure_project(c: httpx.Client) -> int:
    """Find our project by title, or create it. Re-runs must not make a second one."""
    r = c.get("/api/projects", params={"title": PROJECT_TITLE})
    r.raise_for_status()
    for p in r.json().get("results", []):
        if p["title"] == PROJECT_TITLE:
            return p["id"]

    r = c.post("/api/projects", json={
        "title": PROJECT_TITLE,
        "description": "Transactions the fraud model scored into the uncertain band.",
        "label_config": CONFIG.read_text(),
        # One annotation per task is enough here; agreement between analysts is a Part 3 topic.
        "maximum_annotations": 1,
    })
    r.raise_for_status()
    return r.json()["id"]


def ensure_webhook(c: httpx.Client, project_id: int) -> None:
    r = c.get("/api/webhooks/", params={"project": project_id})
    r.raise_for_status()
    if any(w["url"] == WEBHOOK_URL for w in r.json()):
        return
    c.post("/api/webhooks/", json={
        "project": project_id,
        "url": WEBHOOK_URL,
        "send_payload": True,
        "send_for_all_actions": False,
        # UPDATED as well as CREATED: an analyst correcting their own verdict must overwrite
        # the label, not be ignored.
        "actions": ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"],
        "is_active": True,
    }).raise_for_status()


# ------------------------------------------------------------------- push

def _task_data(row) -> dict:
    """One scored transaction -> the fields label_config.xml renders."""
    scored_id, prob, version, p = row
    when = str(p.get("trans_date_trans_time", ""))
    return {
        # Carried through so the webhook can identify the row even if review_tasks is lost.
        "scored_id": scored_id,
        "trans_num": p.get("trans_num"),
        "amt": f"${float(p.get('amt', 0)):,.2f}",
        "category": p.get("category", ""),
        "merchant": str(p.get("merchant", "")).removeprefix("fraud_"),
        "when": when,
        "home": f"{p.get('state', '')} ({p.get('lat')}, {p.get('long')})",
        "merch_at": f"({p.get('merch_lat')}, {p.get('merch_long')})",
        "distance_km": _distance(p),
        "gender": p.get("gender", ""),
        "dob": p.get("dob", ""),
    }


def _distance(p: dict) -> str:
    """Straight-line km, computed here rather than imported from features.py.

    Importing it would give this service an opinion about the model's inputs. It needs a
    number for a human to read, not the model's feature, so eight lines of trigonometry
    cost less than that dependency would.
    """
    from math import asin, cos, radians, sin, sqrt
    try:
        lat1, lon1 = radians(float(p["lat"])), radians(float(p["long"]))
        lat2, lon2 = radians(float(p["merch_lat"])), radians(float(p["merch_long"]))
    except (KeyError, TypeError, ValueError):
        return "?"
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return f"{2 * 6371.0 * asin(sqrt(a)):.0f}"


def push_tasks() -> int:
    """Hand every un-pushed review-band transaction to Label Studio. Returns how many."""
    with _connect() as db:
        rows = db.execute(
            """SELECT s.id, s.probability, s.model_version, s.payload
                 FROM scored_transactions s
                 LEFT JOIN review_tasks r ON r.scored_id = s.id
                WHERE s.decision = 'review' AND r.scored_id IS NULL
                -- Highest suspicion first: if the analysts only get through half the queue,
                -- the half they did is the half worth doing.
                ORDER BY s.probability DESC
                LIMIT %s""", (PUSH_BATCH,)).fetchall()

        if not rows:
            return 0

        pid = STATE["project_id"]
        n = 0
        with _client() as c:
            for row in rows:
                # One task per request rather than a bulk /import: the response carries the
                # task id, which is what lets a webhook be traced back to a score.
                r = c.post("/api/tasks/", json={"project": pid, "data": _task_data(row)})
                r.raise_for_status()
                db.execute(
                    """INSERT INTO review_tasks
                         (scored_id, ls_task_id, ls_project_id, probability, model_version)
                       VALUES (%s,%s,%s,%s,%s) ON CONFLICT (scored_id) DO NOTHING""",
                    (row[0], r.json()["id"], pid, row[1], row[2]))
                n += 1
    STATE["pushed"] += n
    STATE["last_push"] = datetime.now(timezone.utc).isoformat()
    return n


async def push_loop():
    while True:
        try:
            if n := await asyncio.to_thread(push_tasks):
                print(f"[sync] pushed {n} task(s) to Label Studio", flush=True)
            STATE["last_error"] = None
        except Exception as e:                          # noqa: BLE001 -- keep the loop alive
            STATE["last_error"] = f"{type(e).__name__}: {e}"[:300]
            print(f"[sync] push failed: {STATE['last_error']}", flush=True)
        await asyncio.sleep(PUSH_INTERVAL)


# ------------------------------------------------------------------- back

def _choice(result: list) -> str | None:
    for r in result or []:
        choices = (r.get("value") or {}).get("choices") or []
        if choices:
            return choices[0]
    return None


def save_verdict(ls_task_id: int, choice: str, annotator) -> bool:
    """Write one analyst decision. Keyed on the score, so a re-annotation overwrites."""
    verdict = VERDICT_VALUES.get(choice)
    if verdict is None:
        return False
    with _connect() as db:
        row = db.execute(
            "SELECT scored_id, model_version FROM review_tasks WHERE ls_task_id = %s",
            (ls_task_id,)).fetchone()
        if row is None:
            # A task created directly in the Label Studio UI, with no score behind it. Nothing
            # to attach the label to, so it is dropped rather than invented.
            return False
        db.execute(
            """INSERT INTO verdicts (scored_id, verdict, annotator, ls_task_id, flagged_by)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT (scored_id) DO UPDATE SET
                 verdict=EXCLUDED.verdict, decided_at=now(),
                 annotator=EXCLUDED.annotator""",
            (row[0], verdict, str(annotator) if annotator is not None else None,
             ls_task_id, row[1]))
    STATE["verdicts"] += 1
    return True


def pull_annotations() -> dict:
    """Reconcile from the API instead of waiting for webhooks.

    Webhooks are the fast path and this is the correct one: if this container was down, or a
    delivery failed, the annotation still exists in Label Studio and nothing else would ever
    notice. Run it after any outage.

    One export call, not one request per task. The queue holds every review-band transaction
    ever pushed -- tens of thousands after a few sessions -- while the annotated ones are the
    handful an analyst got through. `download_all_tasks=false` returns exactly those, so this
    stays a second's work no matter how long the backlog has grown.
    """
    saved = seen = 0
    with _client() as c:
        r = c.get(f"/api/projects/{STATE['project_id']}/export",
                  params={"exportType": "JSON", "download_all_tasks": "false"})
        r.raise_for_status()
        tasks = r.json()
    for t in tasks:
        for a in t.get("annotations") or []:
            seen += 1
            if (ch := _choice(a.get("result"))) and save_verdict(
                    t["id"], ch, a.get("completed_by")):
                saved += 1
    return {"annotations_seen": seen, "verdicts_written": saved}


# -------------------------------------------------------------------- app

@asynccontextmanager
async def lifespan(app):
    with _connect() as db:
        db.execute(SCHEMA.read_text())

    # Label Studio runs Django migrations on first boot and is slow to answer; the push loop
    # would otherwise spend its first minutes logging connection errors.
    for attempt in range(60):
        try:
            with _client() as c:
                STATE["project_id"] = ensure_project(c)
                ensure_webhook(c, STATE["project_id"])
            print(f"[sync] project {STATE['project_id']} ready, webhook -> {WEBHOOK_URL}", flush=True)
            break
        except Exception as e:                          # noqa: BLE001
            STATE["last_error"] = f"{type(e).__name__}: {e}"[:300]
            # Printed every time, not only on the last attempt: a genuine misconfiguration
            # (bad token, rejected webhook URL) otherwise looks identical to Label Studio
            # still booting, for five silent minutes.
            print(f"[sync] waiting for Label Studio ({attempt + 1}/60): "
                  f"{STATE['last_error']}", flush=True)
            await asyncio.sleep(5)
    else:
        print("[sync] Label Studio never became reachable; /push will keep retrying", flush=True)

    task = asyncio.create_task(push_loop())
    yield
    task.cancel()


app = FastAPI(title="Label sync", lifespan=lifespan)


@app.get("/health")
def health():
    with _connect() as db:
        pending, done = db.execute(
            """SELECT count(*) FILTER (WHERE r.scored_id IS NULL),
                      (SELECT count(*) FROM verdicts)
                 FROM scored_transactions s
                 LEFT JOIN review_tasks r ON r.scored_id = s.id
                WHERE s.decision = 'review'""").fetchone()
    return {**STATE, "label_studio": LS_URL, "unpushed_review_band": pending, "verdicts_total": done}


@app.post("/push")
def push_now():
    """Force a push instead of waiting for the loop -- what the notebook calls."""
    if STATE["project_id"] is None:
        with _client() as c:
            STATE["project_id"] = ensure_project(c)
            ensure_webhook(c, STATE["project_id"])
    return {"pushed": push_tasks(), "project_id": STATE["project_id"]}


@app.post("/pull")
def pull_now():
    return pull_annotations()


@app.post("/webhook")
async def webhook(request: Request):
    """Label Studio's callback. Returns 200 for anything it sends, including what we ignore.

    A non-2xx here makes Label Studio treat the delivery as failed, and there is nothing it
    could usefully retry when the payload is simply not an annotation we track.
    """
    body = await request.json()
    action = body.get("action")
    ann = body.get("annotation") or {}
    if action not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED") or not ann:
        return {"ignored": action}

    # `task` is the id in the annotation payload; when Label Studio inlines the object
    # instead, take its id.
    task = ann.get("task")
    task_id = task.get("id") if isinstance(task, dict) else task
    choice = _choice(ann.get("result"))
    if task_id is None or choice is None:
        return {"ignored": "no task id or no choice"}

    ok = save_verdict(int(task_id), choice, ann.get("completed_by"))
    print(f"[sync] {action} task={task_id} choice={choice} saved={ok}", flush=True)
    return {"saved": ok, "task": task_id, "choice": choice}
