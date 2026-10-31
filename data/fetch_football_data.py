"""Download one real season from football-data.co.uk for your own, private use.

football-data.co.uk states that its data "is intended for private individuals only, NOT commerical or
data training products using automated bots/scrapers/AI". This repository therefore does not
redistribute the file: this script downloads the 2023/24 English Premier League season (E0) to
data/raw/ (git-ignored) and refuses it unless it matches the SHA-256 pinned below, so every run of the
real-data audit reads exactly the bytes the README describes.

    python data/fetch_football_data.py
    python -m sbq --csv data/raw/E0_2324.csv --out data/raw/RESULTS_E0_2324.md
"""

from __future__ import annotations

import hashlib
import pathlib
import urllib.request

URL = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
SHA256 = "b2e057b0ed959f198b0f63d2391c01239f3608e6de5db68edab3f88e04d07ff3"  # fetched 2026-10-31, 380 matches
OUT = pathlib.Path(__file__).resolve().parent / "raw" / "E0_2324.csv"


def verify(payload: bytes, expected: str = SHA256) -> None:
    got = hashlib.sha256(payload).hexdigest()
    if got != expected:
        raise ValueError(f"checksum mismatch: expected {expected}, got {got}; the upstream file changed")


def main() -> None:
    request = urllib.request.Request(URL, headers={"User-Agent": "sports-betting-quant-proof"})
    with urllib.request.urlopen(request, timeout=60) as resp:
        payload = resp.read()
    verify(payload)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(payload)
    print(f"wrote {OUT} ({len(payload)} bytes, sha256 ok)")


if __name__ == "__main__":
    main()
