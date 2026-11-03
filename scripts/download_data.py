"""Download the DAVIS and KIBA source files into ``data/``.

Run once per checkout::

    python scripts/download_data.py

Files come from the DeepDTA repository, which publishes DAVIS as **raw Kd in
nM** (the Kaggle mirror is already pKd - mixing the two double-converts the
scores). Each file is checked against a recorded SHA-256 so everyone in the
group provably works from identical bytes.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# sha256 of each file as downloaded on 2026-11-03.
CHECKSUMS = {
    "davis/proteins.txt": "187cedeee10cd3b58af915a2861a5ed0c4ff42a2d728f4dea5e0b47e3d75b291",
    "davis/ligands_can.txt": "141094d99510a52091d2ea1b05be7c1f66b2c5fcaeeeb261eaf4cbf2128bde6d",
    "davis/Y": "9855c4f234ec6dd18295b0e0cd6ee54214cafa908c6eb1d447b47d454fc0294a",
    "kiba/proteins.txt": "2ba9dbc2959d88b0740aaf28828c4c611c2dbd1c2019547eb2ec1c3daa192604",
    "kiba/ligands_can.txt": "a95797abaf31b17bd2471dbf3e3476f588d5a79797903cc156528f66e55c963b",
    "kiba/Y": "53857b84218ba3d60293c70ab79e2abd83d615d06218c73c150f6dfb4c7134a6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(rel_path: str, expected: str, force: bool = False) -> bool:
    """Download one file unless it is already present and intact."""
    dest = DATA_DIR / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if sha256(dest) == expected:
            print(f"  have  {rel_path}")
            return True
        print(f"  stale {rel_path} (checksum mismatch, re-downloading)")

    url = f"{BASE_URL}/{rel_path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  FAIL  {rel_path}: {exc}")
        return False

    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(payload)
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink()
        print(f"  FAIL  {rel_path}: expected sha256 {expected}, got {actual}")
        return False

    tmp.replace(dest)
    print(f"  got   {rel_path} ({len(payload):,} bytes)")
    return True


def main(argv: list[str]) -> int:
    force = "--force" in argv
    print(f"downloading into {DATA_DIR}")
    ok = [fetch(rel, digest, force) for rel, digest in CHECKSUMS.items()]
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

    if not all(ok):
        print(f"\n{ok.count(False)} file(s) failed. Re-run, or fetch them by hand from")
        print("https://github.com/hkmztrk/DeepDTA/tree/master/data")
        return 1
    print("\nall files present and verified. Next: python src/data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
