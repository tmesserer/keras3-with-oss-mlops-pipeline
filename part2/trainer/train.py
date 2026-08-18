"""B1-B2 · One fraud run, one model. Config from env vars, exactly like Part 1's trainer.

The difference from Part 1 is the metric. At a 0.58% fraud rate, accuracy is a lie: predicting
"never fraud" scores 99.42%. So accuracy IS logged -- to make that visible -- but the champion
metric is PR-AUC (average precision), and the run also logs the operating threshold the
serving layer should use, because a fraud model's output is a decision under a threshold, not
a label.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import keras
from keras import layers
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

import features

EXPERIMENT = os.getenv("EXPERIMENT", "fraud-keras")
REG_NAME = os.getenv("REGISTERED_MODEL", "fraud-keras")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data/fraud"))
# The DVC-versioned snapshot built by part2/pipeline/build_dataset.py. Used when present,
# fraudTrain.csv when not, so this trainer still runs on a clone that has never labelled
# anything. Set SNAPSHOT to an empty string to force the CSV path.
SNAPSHOT = Path(s) if (s := os.getenv("SNAPSHOT", "/data/derived/fraud_training.parquet")) else None

CFG = dict(
    name=os.getenv("RUN_NAME", "run"),
    hidden=os.getenv("HIDDEN", "128,64"),      # comma-separated dense widths
    dropout=float(os.getenv("DROPOUT", "0.3")),
    lr=float(os.getenv("LR", "1e-3")),
    class_weight=os.getenv("CLASS_WEIGHT", "balanced").lower(),   # balanced | none
    embed_max=int(os.getenv("EMBED_MAX", "16")),
)
EPOCHS = int(os.getenv("EPOCHS", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2048"))
SUBSET = int(os.getenv("TRAIN_SUBSET", "0"))     # 0 = all 1.3M
VAL_FRACTION = float(os.getenv("VAL_FRACTION", "0.1"))
TARGET_PRECISION = float(os.getenv("TARGET_PRECISION", "0.90"))


def dvc_md5(pointer: Path) -> str | None:
    """Content hash out of a .dvc pointer, parsed by hand -- dvc is a host tool and has no
    business in this image. Mirrors read_md5() in part2/pipeline/build_dataset.py."""
    if not pointer.exists():
        return None
    for line in pointer.read_text().splitlines():
        key, _, value = line.lstrip("- ").partition(":")
        if key.strip() == "md5":
            return value.strip()
    return None


def load_split():
    """fraudTrain/fraudTest are a TIME split -- never shuffle across the boundary.

    The validation slice is the tail of train, again by time, so it simulates
    'score the future with a model fit on the past'. A random split here would leak
    and quietly inflate every number below.

    Returns the three frames plus provenance tags: which data this run learned from, by
    content hash, so a registered model can be traced to exact bytes rather than to a
    filename that means something different next week.
    """
    cols = features.RAW + ["is_fraud", "trans_num"]
    te = pd.read_csv(DATA_DIR / "fraudTest.csv", usecols=cols)

    if SNAPSHOT and SNAPSHOT.exists():
        tr = pd.read_parquet(SNAPSHOT)
        tags = {"dataset": "derived-snapshot", "dataset_path": SNAPSHOT.name,
                "dvc_md5": dvc_md5(SNAPSHOT.with_suffix(SNAPSHOT.suffix + ".dvc")) or "untracked"}
        n_analyst = int((tr["source"] == "analyst").sum()) if "source" in tr else 0

        # The moment training data comes from production, the original test set stops being
        # held out: every analyst row IS a fraudTest row that a human relabelled. Dropping
        # them keeps test/PR-AUC comparable with earlier runs instead of quietly inflating it.
        if n_analyst:
            reviewed = set(tr.loc[tr["source"] == "analyst", "trans_num"])
            before = len(te)
            te = te[~te["trans_num"].isin(reviewed)]
            tags["test_rows_withheld"] = str(before - len(te))
    else:
        tr = pd.read_csv(DATA_DIR / "fraudTrain.csv", usecols=cols)
        tags = {"dataset": "raw-csv", "dataset_path": "fraudTrain.csv", "dvc_md5": "none"}
        n_analyst = 0

    tr = tr.sort_values("trans_date_trans_time", kind="stable")
    if SUBSET:
        tr = tr.tail(SUBSET)          # tail, not head: keep the most recent behaviour
    cut = int(len(tr) * (1 - VAL_FRACTION))
    tags["analyst_rows"] = str(n_analyst)
    return tr.iloc[:cut], tr.iloc[cut:], te, tags


def build_model(train_df):
    hidden = [int(h) for h in CFG["hidden"].split(",") if h]
    inputs, encoded = {}, []

    x = features.build(train_df)

    for name in features.NUMERIC:
        inp = keras.Input(shape=(1,), name=name, dtype="float32")
        norm = layers.Normalization(name=f"norm_{name}")
        norm.adapt(x[name])           # constants bake into the artifact
        inputs[name] = inp
        encoded.append(norm(inp))

    # Already hashed to bucket indices by features.py, so a plain Embedding is all that's
    # needed -- no StringLookup, and therefore no TensorFlow dependency on the torch backend.
    for name, buckets in features.CATEGORICAL.items():
        inp = keras.Input(shape=(1,), name=name, dtype="int32")
        dim = min(CFG["embed_max"], max(2, int(round(buckets ** 0.5))))
        inputs[name] = inp
        encoded.append(layers.Flatten()(
            layers.Embedding(buckets, dim, name=f"embed_{name}")(inp)))

    h = layers.Concatenate()(encoded)
    for i, units in enumerate(hidden):
        h = layers.Dense(units, activation="relu", name=f"dense_{i}")(h)
        if CFG["dropout"]:
            h = layers.Dropout(CFG["dropout"], name=f"drop_{i}")(h)
    out = layers.Dense(1, activation="sigmoid", name="fraud_probability")(h)

    model = keras.Model(inputs, out, name="fraud_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(CFG["lr"]),
        loss="binary_crossentropy",
        # PR-AUC is the one that matters here; the rest are logged for contrast.
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"),
                 keras.metrics.AUC(curve="ROC", name="roc_auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


def pick_threshold(y_true, y_score):
    """Lowest threshold that still holds TARGET_PRECISION, plus the best-F1 threshold.

    This is the number the application consumes. Choosing it here -- against validation
    data, not test -- keeps the serving layer from inventing its own.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more point than thresholds; drop the last.
    prec, rec = prec[:-1], rec[:-1]
    ok = prec >= TARGET_PRECISION
    at_precision = float(thr[ok][np.argmax(rec[ok])]) if ok.any() else None

    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    return {"threshold_at_precision": at_precision,
            "threshold_best_f1": float(thr[np.argmax(f1)]),
            "best_f1": float(np.max(f1))}


def evaluate(model, df, prefix, extra=None):
    x = features.build(df)
    y = df["is_fraud"].to_numpy("int32")
    p = model.predict(x, batch_size=BATCH_SIZE, verbose=0).ravel()

    m = {f"{prefix}_pr_auc": float(average_precision_score(y, p)),
         f"{prefix}_roc_auc": float(roc_auc_score(y, p)),
         f"{prefix}_base_rate": float(y.mean())}

    if extra is not None:                     # apply the chosen operating point
        pred = (p >= extra).astype("int32")
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        m[f"{prefix}_precision_at_threshold"] = tp / (tp + fp) if tp + fp else 0.0
        m[f"{prefix}_recall_at_threshold"] = tp / (tp + fn) if tp + fn else 0.0
        m[f"{prefix}_alerts"] = float(pred.sum())
        # The number that makes the imbalance concrete: near-perfect, and useless.
        m[f"{prefix}_accuracy"] = float((pred == y).mean())
    return m, y, p


def main():
    print(f"[keras] {keras.__version__} backend={keras.backend.backend()}  mlflow={mlflow.__version__}")
    print(f"[config] {CFG} epochs={EPOCHS} batch={BATCH_SIZE} subset={SUBSET or 'all'}", flush=True)

    tr, val, te, data_tags = load_split()
    print(f"[data] train={len(tr):,} ({tr.is_fraud.mean()*100:.3f}% fraud)  "
          f"val={len(val):,} ({val.is_fraud.mean()*100:.3f}%)  "
          f"test={len(te):,} ({te.is_fraud.mean()*100:.3f}%)", flush=True)
    print(f"[data] {data_tags}", flush=True)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=CFG["name"]) as run:
        model = build_model(tr)
        mlflow.set_tags({"task": "fraud", "keras_version": keras.__version__,
                         "backend": keras.backend.backend(), **data_tags})
        mlflow.log_params(CFG | {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                                 "train_size": len(tr), "target_precision": TARGET_PRECISION})

        y_tr = tr["is_fraud"].to_numpy("float32")
        cw = None
        if CFG["class_weight"] == "balanced":
            # ~172:1 here. Without it the net converges to "never fraud" and stays there.
            pos = y_tr.sum()
            cw = {0: len(y_tr) / (2 * (len(y_tr) - pos)), 1: len(y_tr) / (2 * pos)}
            print(f"[weights] {cw}", flush=True)

        t0 = time.time()
        model.fit(features.build(tr), y_tr,
                  validation_data=(features.build(val), val["is_fraud"].to_numpy("float32")),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=cw, verbose=2,
                  callbacks=[keras.callbacks.LambdaCallback(
                      on_epoch_end=lambda e, logs: mlflow.log_metrics(
                          {k: float(v) for k, v in (logs or {}).items()}, step=e))])
        train_seconds = time.time() - t0

        # Threshold from validation, then judged on test -- never the other way round.
        _, y_val, p_val = evaluate(model, val, "val")
        thresholds = pick_threshold(y_val, p_val)
        chosen = thresholds["threshold_at_precision"] or thresholds["threshold_best_f1"]
        print(f"[threshold] {json.dumps(thresholds)}  -> using {chosen:.6f}", flush=True)

        val_m, _, _ = evaluate(model, val, "val", chosen)
        test_m, _, _ = evaluate(model, te, "test", chosen)
        mlflow.log_metrics(val_m | test_m | {"train_seconds": train_seconds,
                                             "n_params": int(model.count_params()),
                                             **{k: v for k, v in thresholds.items() if v is not None}})
        # The serving layer reads this off the model version rather than guessing.
        mlflow.set_tag("operating_threshold", f"{chosen:.6f}")

        sample = tr.head(2)
        signature = infer_signature(features.build(sample),
                                    model.predict(features.build(sample), verbose=0))
        mlflow.keras.log_model(model, name="model", registered_model_name=REG_NAME,
                               signature=signature)

        print(f"\n--> test_pr_auc={test_m['test_pr_auc']:.4f} "
              f"(base rate {test_m['test_base_rate']:.5f})  "
              f"precision={test_m['test_precision_at_threshold']:.3f} "
              f"recall={test_m['test_recall_at_threshold']:.3f}  "
              f"accuracy={test_m['test_accuracy']:.5f} <- meaningless, see base rate")
        print(f"[done] run '{CFG['name']}' -> experiment '{EXPERIMENT}', model '{REG_NAME}'"
              f"  run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
