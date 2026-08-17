"""Keep the review queue moving: push the review band out, pull analyst verdicts back.

label-sync already sweeps on its own timer (PUSH_INTERVAL). This DAG exists for the other
direction: `POST /pull` re-reads annotations from Label Studio's API and is what recovers
verdicts whose webhook delivery was lost while label-sync was down. Nothing schedules that
by itself, and a verdict that never arrives is a label bought and thrown away.

Both endpoints upsert, so a run that overlaps a webhook delivery cannot double-count.
"""
from __future__ import annotations

import os

import pendulum
import requests
from airflow.sdk import dag, task

SYNC_URL = os.getenv("LABEL_SYNC_URL", "http://label-sync:8002")


@dag(
    dag_id="sync_labels",
    description="push review-band transactions to Label Studio, pull verdicts into Postgres",
    schedule="*/15 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,                      # a missed quarter-hour has nothing to replay: both calls read current state
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=1)},
    tags=["part2", "labels"],
)
def sync_labels():

    @task
    def push() -> dict:
        r = requests.post(f"{SYNC_URL}/push", timeout=120)
        r.raise_for_status()
        out = r.json()
        print(f"[push] {out.get('pushed')} new tasks (project {out.get('project_id')})")
        return out

    @task
    def pull() -> dict:
        r = requests.post(f"{SYNC_URL}/pull", timeout=120)
        r.raise_for_status()
        out = r.json()
        print(f"[pull] {out}")
        return out

    @task
    def report() -> dict:
        """Queue depth in the logs, so a backlog is visible without opening two UIs."""
        r = requests.get(f"{SYNC_URL}/health", timeout=30)
        r.raise_for_status()
        h = r.json()
        print(f"[queue] unpushed review band={h['unpushed_review_band']}  "
              f"verdicts total={h['verdicts_total']}")
        return h

    push() >> pull() >> report()


sync_labels()
