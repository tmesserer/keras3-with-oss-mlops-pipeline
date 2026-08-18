"""G1 · The whole loop, unattended: verify snapshot -> train -> compare -> promote -> reload.

Every task here wraps something run by hand in sections A-F of part2/notebooks/hub.ipynb.
That is the point of doing Airflow last: nothing below is a new capability, it is the same
commands with the waiting, the ordering and the "is the new one actually better" check made
explicit. This DAG is loop ③ in that notebook's diagram.

Two deliberate limits:

  * Building the snapshot is NOT a task. `dvc add` needs the git repo, and a pipeline that
    rewrites a pointer nobody reviewed would version data by accident. So the DAG *verifies*
    the parquet still matches the committed .dvc pointer and fails loudly if it does not --
    rebuilding stays `python part2/pipeline/build_dataset.py` on the host.
  * Promotion is a comparison, not a rubber stamp. A run that does not beat the incumbent
    on test PR-AUC skips the promote/reload half, and the champion stays where it is.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pendulum
import requests
from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowFailException, AirflowSkipException
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

MLFLOW = os.environ["MLFLOW_TRACKING_URI"]
REG_NAME = os.getenv("REGISTERED_MODEL", "fraud-keras")
SERVE_URL = os.getenv("SERVE_URL", "http://fraud-serve:8001")

SNAPSHOT = Path(os.getenv("SNAPSHOT", "/data/derived/fraud_training.parquet"))
POINTER = SNAPSHOT.with_suffix(SNAPSHOT.suffix + ".dvc")

# Paths as the DOCKER DAEMON sees them. Airflow's own /data and /mlruns mounts are useless
# to a sibling container: the daemon resolves bind sources on the host, not in this
# container's namespace. Hence the host path, passed in by compose.
HOST_REPO = os.environ["HOST_REPO_DIR"]
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "keras-mlops_default")
TRAINER_IMAGE = os.getenv("TRAINER_IMAGE", "keras-mlops-fraud-trainer:latest")

# Restated from compose.part2.yml's fraud-trainer block, which stays the source of truth --
# there is no way to ask compose for a service's environment from inside a DAG. Only the
# hyperparameters below are meant to differ between a hand-run and a scheduled one.
TRAINER_ENV = {
    "MLFLOW_TRACKING_URI": MLFLOW,
    "KERAS_BACKEND": "torch",
    "KERAS_HOME": "/tmp/.keras",
    "HOME": "/tmp",
    "USER": "mlflow",
    "GIT_PYTHON_REFRESH": "quiet",
    "EXPERIMENT": "fraud-keras",
    "REGISTERED_MODEL": REG_NAME,
    "DATA_DIR": "/data/fraud",
    "SNAPSHOT": str(SNAPSHOT),
    # Hyperparameters, and the only part meant to vary between runs. `environment` is a
    # templated field, so a trigger can override them for one run without touching compose:
    #   airflow dags trigger fraud_retrain --conf '{"epochs": "12", "hidden": "256,128"}'
    "EPOCHS": "{{ dag_run.conf.get('epochs', params.epochs) }}",
    "HIDDEN": "{{ dag_run.conf.get('hidden', params.hidden) }}",
    "TRAIN_SUBSET": "{{ dag_run.conf.get('train_subset', params.train_subset) }}",
    "TARGET_PRECISION": "{{ dag_run.conf.get('target_precision', params.target_precision) }}",
}


def _client():
    from mlflow import MlflowClient

    return MlflowClient(tracking_uri=MLFLOW)


def _artifact_dir(client, mv) -> str:
    """Where a model version's files actually sit under /mlruns.

    MLflow 3 registers versions with source `models:/m-<logged_model_id>`, an indirection --
    `storage_location` comes back None and the path only appears on the logged model behind
    it. Older versions still carry a plain file:// URI, hence both branches.
    """
    if mv.source.startswith("models:/m-"):
        location = client.get_logged_model(mv.source.rsplit("/", 1)[-1]).artifact_location
    else:
        location = mv.source
    return location.removeprefix("file://")


def _pointer_md5() -> str:
    """The `md5:` field of the .dvc pointer. Third copy of this parser (build_dataset.py,
    train.py, here) -- each lives in an image that must not depend on the other two."""
    for line in POINTER.read_text().splitlines():
        key, _, value = line.lstrip("- ").partition(":")
        if key.strip() == "md5":
            return value.strip()
    raise AirflowFailException(f"no md5 field in {POINTER}")


@dag(
    dag_id="fraud_retrain",
    description="verify snapshot -> train -> compare -> promote @champion -> reload serving",
    schedule=None,                      # triggered by hand; a fraud model does not retrain on a clock
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,                  # two trainers writing the same mlruns is not a scenario worth having
    default_args={"retries": 0},        # a failed train should be read, not retried blindly
    params={                            # defaults for the env overrides above; --conf wins over these
        "epochs": os.getenv("DAG_EPOCHS", "5"),
        "hidden": os.getenv("DAG_HIDDEN", "128,64"),
        "train_subset": os.getenv("DAG_TRAIN_SUBSET", "0"),
        "target_precision": os.getenv("DAG_TARGET_PRECISION", "0.90"),
    },
    tags=["part2", "fraud"],
)
def fraud_retrain():

    @task
    def check_snapshot() -> str:
        """The parquet on disk must be the bytes the committed pointer names.

        Silently training on a snapshot that no longer matches its pointer produces a model
        tagged with a hash that cannot reproduce it -- worse than no versioning at all,
        because it looks versioned.
        """
        if not SNAPSHOT.exists() or not POINTER.exists():
            raise AirflowFailException(
                f"{SNAPSHOT.name} or its .dvc pointer is missing. Build it on the host: "
                "python part2/pipeline/build_dataset.py")

        h = hashlib.md5()
        with SNAPSHOT.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual, expected = h.hexdigest(), _pointer_md5()

        if actual != expected:
            raise AirflowFailException(
                f"snapshot md5 {actual} != pointer {expected}. The parquet was rebuilt and "
                "the pointer not committed (or the reverse). Run "
                "`python part2/pipeline/build_dataset.py` on the host and commit "
                f"{POINTER.name}.")

        print(f"[snapshot] {SNAPSHOT.name} matches {POINTER.name}  md5={actual}")
        return actual

    train = DockerOperator(
        task_id="train",
        # The same image, the same entrypoint, the same env-var-only config as
        # `docker compose --profile tools2 run --rm fraud-trainer`. Airflow supplies the
        # run name and waits; it does not know what training is.
        image=TRAINER_IMAGE,
        # run_after, not ds/ts_nodash: those come from logical_date, and an Airflow 3 manual
        # run of an unscheduled DAG has none -- the template renders 'ds is undefined' and the
        # task fails before docker is ever contacted.
        container_name="fraud-trainer-airflow-{{ dag_run.run_after.strftime('%Y%m%d-%H%M%S') }}",
        environment={**TRAINER_ENV,
                     "RUN_NAME": "airflow-{{ dag_run.run_after.strftime('%Y%m%d-%H%M') }}"},
        user=f"{os.getenv('DOCKER_UID', '1000')}:{os.getenv('DOCKER_GID', '1000')}",
        network_mode=DOCKER_NETWORK,    # mlflow-server is a service name, not a host port
        mounts=[
            Mount(source=f"{HOST_REPO}/mlruns", target="/mlruns", type="bind"),
            Mount(source=f"{HOST_REPO}/data", target="/data", type="bind", read_only=True),
        ],
        mount_tmp_dir=False,            # the default /tmp bind is a docker-in-docker trap: the source path does not exist on the host
        auto_remove="success",          # keep the container on failure, so its logs survive the DAG run
        docker_url="unix://var/run/docker.sock",
        api_version="auto",
        retrieve_output=False,
    )

    @task
    def promote(snapshot_md5: str) -> dict:
        """Move @champion only if the new version wins on test PR-AUC.

        PR-AUC, not accuracy: at a 0.4% base rate accuracy ranks a model that predicts
        'never fraud' above every useful one. See part2/trainer/train.py.
        """
        client = _client()
        versions = client.search_model_versions(f"name='{REG_NAME}'")
        candidate = max(versions, key=lambda v: int(v.version))
        run = client.get_run(candidate.run_id)

        # The version that just appeared must be the one this DAG produced. If it was
        # trained from other bytes, someone registered a model mid-run and promoting it
        # would attribute their work to this snapshot.
        if run.data.tags.get("dvc_md5") != snapshot_md5:
            raise AirflowFailException(
                f"v{candidate.version} was trained from dvc_md5 "
                f"{run.data.tags.get('dvc_md5')}, not {snapshot_md5}")

        # Registry rows outlive artifacts here (see CLAUDE.md): a dangling @champion fails
        # at fraud-serve STARTUP, long after this DAG has gone green. Check now instead.
        artifact = _artifact_dir(client, candidate)
        if not Path(artifact).exists():
            raise AirflowFailException(
                f"v{candidate.version} has no artifacts on disk at {artifact}; promoting it "
                "would break fraud-serve on its next restart")

        score = run.data.metrics.get("test_pr_auc")
        if score is None:
            raise AirflowFailException(f"v{candidate.version} logged no test_pr_auc")

        try:
            current = client.get_model_version_by_alias(REG_NAME, "champion")
            incumbent = client.get_run(current.run_id).data.metrics.get("test_pr_auc", 0.0)
        except Exception:
            current, incumbent = None, None      # nothing holds the alias yet

        print(f"[promote] candidate v{candidate.version} test_pr_auc={score:.4f}  "
              f"champion={'v' + current.version if current else 'none'} "
              f"test_pr_auc={incumbent if incumbent is None else round(incumbent, 4)}")

        if incumbent is not None and score <= incumbent:
            raise AirflowSkipException(
                f"v{candidate.version} ({score:.4f}) does not beat v{current.version} "
                f"({incumbent:.4f}); champion unchanged")

        client.set_registered_model_alias(REG_NAME, "champion", candidate.version)
        print(f"[promote] @champion -> v{candidate.version}")
        return {"version": candidate.version, "test_pr_auc": score}

    @task
    def reload_serving(promoted: dict) -> dict:
        """Serving resolves @champion once at startup, so promotion alone changes nothing.

        This is the moment the app's decision mix moves -- the endpoint swaps models under a
        stream it never stops serving.
        """
        r = requests.post(f"{SERVE_URL}/reload", timeout=180)
        r.raise_for_status()
        health = r.json()
        if str(health.get("version")) != str(promoted["version"]):
            raise AirflowFailException(
                f"fraud-serve reloaded v{health.get('version')}, expected "
                f"v{promoted['version']}")
        print(f"[reload] serving v{health['version']} "
              f"decline>={health['threshold_decline']:.6f}")
        return health

    md5 = check_snapshot()
    promoted = promote(md5)
    # promote() already consumes md5, but DockerOperator returns nothing to consume, so the
    # train -> promote edge has to be drawn by hand or Airflow would run them in parallel.
    md5 >> train >> promoted
    reload_serving(promoted)


fraud_retrain()
