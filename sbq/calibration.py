"""Proper scores and a reliability table for probability forecasts of k-outcome markets.

`probs` is an (n, k) array, one row per match, each row summing to one; `outcome` holds the index
of what happened (0 = home, 1 = draw, 2 = away for 1X2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

LOG_FLOOR = 1e-15


def _check(probs, outcome) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcome, dtype=int)
    if p.ndim != 2 or y.ndim != 1 or p.shape[0] != y.size:
        raise ValueError("probs must be (n, k) and outcome length n")
    if ((y < 0) | (y >= p.shape[1])).any():
        raise ValueError("outcome index out of range")
    if (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of probs must be non-negative and sum to 1")
    return p, y


def brier(probs, outcome) -> float:
    """Mean over matches of the squared error summed over outcomes (0 is perfect, 2 the worst)."""
    p, y = _check(probs, outcome)
    hit = np.eye(p.shape[1])[y]
    return float(((p - hit) ** 2).sum(axis=1).mean())


def log_loss(probs, outcome) -> float:
    """Mean negative log probability given to what happened, floored at 1e-15."""
    p, y = _check(probs, outcome)
    return float(-np.log(np.maximum(p[np.arange(y.size), y], LOG_FLOOR)).mean())


def reliability_table(probs, outcome, bins: int = 10) -> pd.DataFrame:
    """Every (match, outcome) forecast binned by its probability: does 30% happen 30% of the time?

    Empty bins are left out. A probability of exactly 1 goes in the top bin.
    """
    p, y = _check(probs, outcome)
    forecast = p.ravel()
    happened = np.eye(p.shape[1])[y].ravel()
    idx = np.minimum((forecast * bins).astype(int), bins - 1)
    rows = []
    for b in np.unique(idx):
        mask = idx == b
        rows.append((b / bins, (b + 1) / bins, int(mask.sum()), forecast[mask].mean(), happened[mask].mean()))
    return pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "n", "mean_pred", "observed"])


def scorecard(forecasters: Mapping[str, Sequence], outcome) -> pd.DataFrame:
    """Brier and log loss per named forecaster on the same matches, in the order given."""
    rows = {name: (brier(p, outcome), log_loss(p, outcome), len(outcome)) for name, p in forecasters.items()}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["brier", "log_loss", "n"])
