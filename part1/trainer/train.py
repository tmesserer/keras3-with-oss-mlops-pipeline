"""B3 · One run, one model. Every hyperparameter arrives as an env var from the container launch."""
import os, time
from pathlib import Path

import numpy as np
import keras
from keras import layers
import mlflow
import mlflow.keras  # explicit: registers the Keras 3 flavor
from mlflow.models import infer_signature

EXPERIMENT = os.getenv("EXPERIMENT", "mnist-keras")
REG_NAME   = os.getenv("REGISTERED_MODEL", "mnist-keras")
DATA_FILE  = Path("/data/mnist.npz")

CFG = dict(
    name    = os.getenv("RUN_NAME", "run"),
    kind    = os.getenv("KIND", "mlp").lower(),      # mlp | cnn
    size    = int(os.getenv("SIZE", "128")),         # mlp hidden units / cnn conv filters
    dropout = float(os.getenv("DROPOUT", "0.2")),
    opt     = os.getenv("OPT", "adam").lower(),      # adam | sgd
    lr      = float(os.getenv("LR", "1e-3")),
)
EPOCHS     = int(os.getenv("EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
SUBSET     = int(os.getenv("TRAIN_SUBSET", "20000"))     # 0 = all 60k


def load_data():
    """Cache MNIST into the mounted /data folder so reruns are offline."""
    if DATA_FILE.exists():
        d = np.load(DATA_FILE)
        (x_tr, y_tr), (x_te, y_te) = (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])
        print(f"[data] loaded {DATA_FILE}")
    else:
        (x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA_FILE, x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
        print(f"[data] downloaded and cached to {DATA_FILE}")
    # Keras 3 is channels_last on every backend, including torch.
    prep = lambda x: (x.astype("float32") / 255.0)[..., None]
    x_tr, x_te = prep(x_tr), prep(x_te)
    if SUBSET:
        x_tr, y_tr = x_tr[:SUBSET], y_tr[:SUBSET]
    print(f"[data] train={x_tr.shape} test={x_te.shape}")
    return (x_tr, y_tr), (x_te, y_te)


def build_model(cfg):
    m = keras.Sequential(name=cfg["name"].replace("-", "_"))
    m.add(keras.Input(shape=(28, 28, 1)))
    if cfg["kind"] == "cnn":
        m.add(layers.Conv2D(cfg["size"], 3, activation="relu", padding="same"))
        m.add(layers.MaxPooling2D())
        m.add(layers.Flatten())
    else:
        m.add(layers.Flatten())
        m.add(layers.Dense(cfg["size"], activation="relu"))
    if cfg["dropout"]:
        m.add(layers.Dropout(cfg["dropout"]))
    m.add(layers.Dense(10, activation="softmax"))
    opt = (keras.optimizers.Adam(cfg["lr"]) if cfg["opt"] == "adam"
           else keras.optimizers.SGD(cfg["lr"], momentum=0.9))
    m.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


class MLflowEpochLogger(keras.callbacks.Callback):
    """The whole Keras->MLflow bridge: log every epoch's metrics as a step."""
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def main():
    print(f"[keras] {keras.__version__} backend={keras.backend.backend()}  mlflow={mlflow.__version__}")
    print(f"[config] {CFG} epochs={EPOCHS} batch_size={BATCH_SIZE} subset={SUBSET}", flush=True)
    (x_tr, y_tr), (x_te, y_te) = load_data()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=CFG["name"]) as run:
        model = build_model(CFG)
        mlflow.set_tags({"kind": CFG["kind"],
                         "keras_version": keras.__version__,
                         "backend": keras.backend.backend()})
        mlflow.log_params(CFG | {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                                 "train_size": len(x_tr)})

        t0 = time.time()
        model.fit(x_tr, y_tr, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=[MLflowEpochLogger()], verbose=2)
        loss, acc = model.evaluate(x_te, y_te, verbose=0)
        mlflow.log_metrics({"test_loss": float(loss), "test_accuracy": float(acc),
                            "train_seconds": time.time() - t0,
                            "n_params": int(model.count_params())})

        # The registry is the only place a model is stored — no parallel copy on the host.
        # signature, not input_example: the keras flavor calls _save_example() before it
        # creates the model dir (still true in 3.15), so input_example raises
        # FileNotFoundError. A signature covers the same need.
        sample = x_te[:1]
        signature = infer_signature(sample, model.predict(sample, verbose=0))
        mlflow.keras.log_model(model, name="model",   # 'artifact_path' is deprecated in mlflow 3
                               registered_model_name=REG_NAME,
                               signature=signature)

        print(f"\n--> test_accuracy={acc:.4f}  run_id={run.info.run_id}", flush=True)

    print(f"[done] run '{CFG['name']}' -> experiment '{EXPERIMENT}', registered model '{REG_NAME}'")


if __name__ == "__main__":
    main()
