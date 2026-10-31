"""The honest run on other synthetic seeds: is the committed seed's ROI an unusual draw?

python scripts/seed_sweep.py   # about a minute; leagues are written to a temporary directory

Prints one line per seed and the ranges quoted in the README. z = (realized ROI - expected ROI under
the truth) / standard error of the realized ROI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sbq import backtest, data, synthetic

SEEDS = [synthetic.SEED, *range(1, 10)]


def main() -> None:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in SEEDS:
            path = Path(tmp) / f"league_{seed}.csv"
            synthetic.write_league(path, seed)
            s = backtest.run(data.load_matches(path), "honest").summary
            z = (s["roi"] - s["true_roi"]) / s["roi_se"]
            rows.append((s["roi"], z, s["true_roi"], s["clv_mean"]))
            print(f"seed {seed:>9}: bets {s['bets']:4d}  realized ROI {s['roi']:+.1%} +/- {s['roi_se']:.1%}  "
                  f"z {z:+.2f}  expected ROI {s['true_roi']:+.1%}  mean CLV {s['clv_mean']:+.1%}")
    roi, z, true_roi, clv_mean = zip(*rows)
    print(f"{len(SEEDS)} seeds: realized ROI {min(roi):+.1%} to {max(roi):+.1%}, z {min(z):+.2f} to {max(z):+.2f}, "
          f"expected ROI {min(true_roi):+.1%} to {max(true_roi):+.1%}, mean CLV {min(clv_mean):+.1%} to "
          f"{max(clv_mean):+.1%} (negative on {sum(c < 0 for c in clv_mean)} of {len(SEEDS)})")


if __name__ == "__main__":
    main()
