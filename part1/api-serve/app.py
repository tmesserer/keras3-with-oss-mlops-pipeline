"""D1 · Serves whichever model version carries the @champion alias in MLflow."""
import json, os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mlflow
import mlflow.keras  # explicit: registers the Keras 3 flavor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_URI = os.getenv("MODEL_URI", "models:/mnist-keras@champion")
INFER_LOG = Path("/inference/predictions.jsonl")
DATA_FILE = Path("/data/mnist.npz")
STATE = {}


def load_champion():
    """models:/<name>@<alias>  ->  concrete version + keras model."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    name, alias = MODEL_URI.removeprefix("models:/").split("@")
    mv = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    STATE.update(model=mlflow.keras.load_model(MODEL_URI), model_name=name, alias=alias,
                 version=mv.version, run_id=mv.run_id,
                 loaded_at=datetime.now(timezone.utc).isoformat())
    print(f"[serve] loaded {name} v{mv.version} (@{alias}) run_id={mv.run_id}", flush=True)


@asynccontextmanager
async def lifespan(app):
    load_champion()
    yield


app = FastAPI(title="MNIST champion server", lifespan=lifespan)


class PredictRequest(BaseModel):
    pixels: list[float]        # 784 values, 0-255 or 0-1
    label: int | None = None   # optional ground truth, just gets logged


def _record(payload: dict) -> dict:
    """Append every inference to the mounted /inference folder, then return it."""
    payload = {"ts": datetime.now(timezone.utc).isoformat(),
               "model": STATE["model_name"], "version": STATE["version"], **payload}
    INFER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with INFER_LOG.open("a") as f:
        f.write(json.dumps(payload) + "\n")
    return payload


def _predict(pixels, label=None, source="request"):
    x = np.asarray(pixels, dtype="float32").reshape(1, 28, 28, 1)
    if x.max() > 1.5:
        x = x / 255.0
    probs = STATE["model"].predict(x, verbose=0)[0]
    return _record({"source": source, "label": label,
                    "prediction": int(np.argmax(probs)),
                    "confidence": round(float(np.max(probs)), 4),
                    "probs": [round(float(p), 4) for p in probs]})


@app.get("/health")
def health():
    return {k: v for k, v in STATE.items() if k != "model"}


@app.post("/reload")
def reload_model():
    load_champion()
    return health()


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.pixels) != 784:
        raise HTTPException(400, f"expected 784 pixels, got {len(req.pixels)}")
    return _predict(req.pixels, req.label)


@app.get("/predict/sample/{i}")
def predict_sample(i: int):
    """Zero-effort demo: curl localhost:8000/predict/sample/0"""
    if not DATA_FILE.exists():
        raise HTTPException(404, "run the trainer first, it caches /data/mnist.npz")
    d = np.load(DATA_FILE)
    if not 0 <= i < len(d["x_test"]):
        raise HTTPException(400, "index out of range")
    return _predict(d["x_test"][i], int(d["y_test"][i]), source=f"mnist_test[{i}]")
