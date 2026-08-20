"""A1 · Fetch the Sparkov fraud dataset into ./data/fraud. Idempotent: does nothing if present.

Runs on the HOST (in the .venv), not in a container -- the trainer mounts ./data and
expects the CSVs to already exist, exactly like Part 1's cached mnist.npz.

Source is a public Google Drive file. Drive refuses to stream files this large without a
confirm token, and a plain curl/wget silently saves the 2KB HTML warning page instead with
exit code 0 -- which then surfaces much later as an unreadable CSV. gdown handles the
token, and _assert_not_html below is the second check.
"""
import io
import sys
import zipfile
from pathlib import Path

DRIVE_ID = "10EoQDn9pEai_vFg4rp9JvPtKtAkx8VWg"
KAGGLE_DS = "kartik2112/fraud-detection"

ROOT = next(d for d in [Path.cwd(), *Path.cwd().parents] if (d / "docker-compose.yml").exists())
DEST = ROOT / "data" / "fraud"
EXPECTED = ("fraudTrain.csv", "fraudTest.csv")


def _found() -> list[Path]:
    return [p for name in EXPECTED if (p := DEST / name).exists()]


def _assert_not_html(blob: bytes, where: str) -> None:
    """Drive's virus-scan warning page is a 200 OK containing HTML. Catch it here."""
    head = blob[:512].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")):
        sys.exit(
            f"[fetch] {where} returned an HTML page, not data ({len(blob)} bytes).\n"
            f"        The Drive confirm-token step failed. Download manually from\n"
            f"        https://drive.google.com/file/d/{DRIVE_ID}/view\n"
            f"        and unzip the CSVs into {DEST}"
        )


def _unpack(blob: bytes) -> None:
    _assert_not_html(blob, "download")
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        # The archive nests the CSVs under a folder; flatten to data/fraud/*.csv.
        for member in z.namelist():
            name = Path(member).name
            if name in EXPECTED:
                (DEST / name).write_bytes(z.read(member))
                print(f"[fetch] wrote {DEST / name}")


def from_drive() -> bool:
    try:
        import gdown
    except ImportError:
        print("[fetch] gdown not installed -- pip install -r requirements-host.txt")
        return False
    buf = io.BytesIO()
    gdown.download(id=DRIVE_ID, output=buf, quiet=False)
    blob = buf.getvalue()
    if not blob:
        return False
    _unpack(blob)
    return True


def from_kaggle() -> bool:
    """Fallback. Needs ~/.kaggle/kaggle.json."""
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        print("[fetch] no ~/.kaggle/kaggle.json, skipping Kaggle fallback")
        return False
    import subprocess
    DEST.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DS, "--unzip", "-p", str(DEST)],
        check=True,
    )
    return True


def main() -> None:
    if len(_found()) == len(EXPECTED):
        print(f"[fetch] already present in {DEST}, nothing to do")
        return
    if not (from_drive() or from_kaggle()):
        sys.exit(f"[fetch] could not obtain the dataset -- see {DEST} instructions above")

    missing = set(EXPECTED) - {p.name for p in _found()}
    if missing:
        sys.exit(f"[fetch] download finished but {sorted(missing)} are missing from {DEST}")
    for p in _found():
        print(f"[fetch] {p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
