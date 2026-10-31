"""Closing line value (CLV): did the price you took beat the market's last word?

CLV of one bet = taken_odds * p_close - 1, where p_close is the closing probability of the
selection with the closing margin removed. It equals taken_odds / fair_closing_odds - 1: the
expected return of the bet if the closing line is the truth. Positive CLV over many bets is the
standard evidence of a real edge, because it is measured against prices, not against noisy results.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sbq.devig import devig


def clv(taken_odds: float, closing_market: Sequence[float], selection: int, method: str = "multiplicative") -> float:
    fair = devig(closing_market, method)
    if not 0 <= selection < fair.size:
        raise ValueError("selection is not an outcome of the closing market")
    return float(taken_odds * fair[selection] - 1.0)


def summarize(values: Sequence[float]) -> dict:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "share_positive": None}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "share_positive": float((v > 0).mean())}
