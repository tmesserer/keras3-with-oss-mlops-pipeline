"""Raw transaction -> model inputs. Shared verbatim by the trainer and the serving image.

This module is deliberately *dumb*: it parses dates and does arithmetic, and it learns
nothing. Every statistical transform (normalisation constants, category vocabularies) lives
inside the Keras model, so the registry artifact stays self-contained.

Why share code at all, when Part 1 made a point of api-serve never importing the trainer?
Because these columns need date parsing and trigonometry that Keras layers can't express, and
the alternative -- reimplementing them in the server -- is the classic training/serving skew
bug. Sharing one file is the small, explicit price.

Note what is NOT here: anything needing history, such as "this card's average spend".
Those are causal features that would need a feature store at serving time, since the endpoint
sees one transaction with no past. Keeping every feature row-local is what lets /score work
from the request body alone.
"""
import zlib

import numpy as np
import pandas as pd

# Columns the model consumes. Order matters only for readability.
NUMERIC = ["amt", "log_amt", "hour", "dayofweek", "age_years", "distance_km", "log_city_pop"]

# Categoricals go through the hashing trick: string -> crc32 -> fixed bucket index.
#
# The obvious choice, keras.layers.StringLookup, requires TensorFlow even on the torch
# backend (it wraps tf.lookup), and pulling a second framework into the image to map ~700
# merchant names is a bad trade. Hashing also keeps this module stateless: there is no
# learned vocabulary that would have to be persisted next to the model and kept in sync
# with it. Buckets are sized well above the observed cardinality to keep collisions rare.
CATEGORICAL = {"category": 32,    # 14 distinct
               "merchant": 2048,  # 693 distinct
               "gender": 4,       # 2 distinct
               "state": 128}      # 51 distinct

# Raw CSV columns needed to build the above (plus the label, added by the trainer).
RAW = ["trans_date_trans_time", "amt", "category", "merchant", "gender", "state",
       "lat", "long", "merch_lat", "merch_long", "city_pop", "dob"]


def _hash_bucket(values: np.ndarray, buckets: int) -> np.ndarray:
    """Stable string -> bucket index.

    crc32, not Python's hash(): hash() is salted per process (PYTHONHASHSEED), so the
    trainer and the server would disagree on every bucket and the model would score noise.
    """
    return np.array([zlib.crc32(str(v).encode()) % buckets for v in values], dtype="int32")


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance cardholder home -> merchant. Fraud tends to be far away."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """DataFrame of raw columns -> dict of arrays keyed by model input name."""
    ts = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])

    out = {
        "amt": df["amt"].to_numpy("float32"),
        # Amounts are heavily right-skewed (median 47, max 28949); the log is what
        # separates the classes, the raw value is kept for outlier magnitude.
        "log_amt": np.log1p(df["amt"].to_numpy("float64")).astype("float32"),
        "hour": ts.dt.hour.to_numpy("float32"),
        "dayofweek": ts.dt.dayofweek.to_numpy("float32"),
        "age_years": ((ts - dob).dt.days / 365.25).to_numpy("float32"),
        "distance_km": _haversine_km(
            df["lat"].to_numpy("float64"), df["long"].to_numpy("float64"),
            df["merch_lat"].to_numpy("float64"), df["merch_long"].to_numpy("float64"),
        ).astype("float32"),
        "log_city_pop": np.log1p(df["city_pop"].to_numpy("float64")).astype("float32"),
    }
    for c, buckets in CATEGORICAL.items():
        out[c] = _hash_bucket(df[c].to_numpy(), buckets)

    return {k: v.reshape(-1, 1) for k, v in out.items()}
