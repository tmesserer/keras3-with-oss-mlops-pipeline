"""F1 · Build the derived training snapshot: original history + what the analysts decided.

Runs on the HOST, like fetch_data.py -- it needs `dvc add`, and dvc needs the git repo.

The output is one parquet containing every labelled row available to the next training run:

  source='original'  the 1.3M rows of fraudTrain.csv, labels that came with the dataset
  source='analyst'   transactions the model sent to review and a human then ruled on

The second group is the only new information in the system. It is also, deliberately, a
*biased* sample: only review-band transactions are ever queued, so every analyst row sits
near the operating threshold. That is where the labels are worth buying -- and it means the
snapshot's base rate is not the population's. `n_analyst` is logged by the trainer so the
effect stays traceable rather than surprising.

`dvc add` turns the parquet into a content hash in a `.dvc` pointer. The pointer is
committed; the parquet is not. That is the whole trick: git carries an exact identity for a
90MB file it never stores, and a model's run tag can name the data that produced it.
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import psycopg

ROOT = next(d for d in [Path.cwd(), *Path.cwd().parents] if (d / "docker-compose.yml").exists())
OUT = ROOT / "data" / "derived" / "fraud_training.parquet"
CSV = ROOT / "data" / "fraud" / "fraudTrain.csv"

# Host-side, so postgres is reached on the published port rather than by service name.
PG = os.getenv("POSTGRES_DSN", "postgresql://mlops:mlops@localhost:5432/mlops")

# The column list the model consumes, imported rather than restated: a snapshot missing a
# column the trainer expects fails at fit time, long after this script has exited happily.
sys.path.insert(0, str(ROOT / "part2" / "common"))
from features import RAW                                                   # noqa: E402

PROVENANCE = ["source", "trans_num", "reviewed_at", "flagged_by"]


def original() -> pd.DataFrame:
    df = pd.read_csv(CSV, usecols=RAW + ["is_fraud", "trans_num"])
    df["source"] = "original"
    df["reviewed_at"] = pd.NaT
    df["flagged_by"] = None
    return df


def reviewed() -> pd.DataFrame:
    """Analyst verdicts, with the transaction exactly as the endpoint saw it.

    The features come out of scored_transactions.payload rather than being re-read from
    fraudTest.csv: the payload is what was actually scored, and a row the analyst judged
    must go into training as the thing they were shown.
    """
    with psycopg.connect(PG) as c:
        rows = c.execute(
            """SELECT s.payload, v.verdict, v.decided_at, v.flagged_by, s.trans_num
                 FROM verdicts v
                 JOIN scored_transactions s ON s.id = v.scored_id
                ORDER BY v.decided_at""").fetchall()

    if not rows:
        return pd.DataFrame(columns=RAW + ["is_fraud"] + PROVENANCE)

    df = pd.DataFrame([{**{k: p.get(k) for k in RAW},
                        "is_fraud": int(verdict),
                        "trans_num": trans_num,
                        "source": "analyst",
                        "reviewed_at": decided_at,
                        "flagged_by": flagged_by}
                       for p, verdict, decided_at, flagged_by, trans_num in rows])
    return df


def build() -> pd.DataFrame:
    # Concatenating an empty frame in would make pandas re-infer every dtype from all-NA
    # columns; before any analyst has ruled on anything, there is simply nothing to add.
    parts = [p for p in (original(), reviewed()) if len(p)]
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

    # An analyst verdict on a transaction that is also in the original file overrides it --
    # the human looked at it more recently than the dataset's author did.
    df = df.drop_duplicates(subset="trans_num", keep="last")

    # Time order, stable, so two builds of the same rows produce byte-identical parquet and
    # therefore the same DVC hash. A snapshot whose hash churns on rebuild versions nothing.
    df = df.sort_values(["trans_date_trans_time", "trans_num"], kind="stable")
    return df[RAW + ["is_fraud"] + PROVENANCE].reset_index(drop=True)


def dvc(*args: str) -> str:
    r = subprocess.run([sys.executable, "-m", "dvc", *args], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"[build] dvc {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def read_md5(pointer: Path) -> str:
    """The `md5:` field of a .dvc pointer, which is YAML but only ever a handful of lines.

    Parsed by hand so the trainer can do the identical read without dvc installed in its
    image -- see part2/trainer/train.py. Note the leading '- ': the field sits inside the
    `outs:` list, so a naive startswith('md5:') finds nothing.
    """
    for line in pointer.read_text().splitlines():
        key, _, value = line.lstrip("- ").partition(":")
        if key.strip() == "md5":
            return value.strip()
    sys.exit(f"[build] no md5 in {pointer}")


def main() -> None:
    if not CSV.exists():
        sys.exit(f"[build] {CSV} missing -- run part2/pipeline/fetch_data.py first")

    df = build()
    n_analyst = int((df["source"] == "analyst").sum())
    print(f"[build] {len(df):,} rows  ({n_analyst:,} analyst, "
          f"{len(df) - n_analyst:,} original)  fraud={df.is_fraud.mean() * 100:.3f}%")
    if n_analyst == 0:
        print("[build] no verdicts yet -- the snapshot is the original data and nothing more. "
              "Annotate in Label Studio (:8080) and rebuild to watch the hash move.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    if not (ROOT / ".dvc").exists():
        dvc("init")
        print("[build] dvc initialised")
    dvc("add", str(OUT.relative_to(ROOT)))

    # The hash lives in the pointer file; the trainer reads the same line and tags its run
    # with it, which is what connects a registered model back to exact bytes.
    pointer = OUT.with_suffix(OUT.suffix + ".dvc")
    md5 = read_md5(pointer)
    print(f"[build] {OUT.relative_to(ROOT)}  {OUT.stat().st_size / 1e6:.1f} MB")
    print(f"[build] dvc md5 {md5}  -> commit {pointer.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
