"""Independent-Poisson match model: goal rates -> scoreline matrix -> market probabilities.

`score_matrix(lh, la)[h, a]` is P(home scores h, away scores a). Everything else reads that matrix:
win/draw/loss, match totals, team totals, expected goals. Independence and a constant rate are the
model's assumptions; they are fine for goals and poor for high-scoring sports (see README, limits).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.stats import poisson as _pois


class OverUnder(NamedTuple):
    over: float
    under: float
    push: float  # non-zero only on whole-number lines, where the stake is returned


def score_matrix(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    for lam in (lam_home, lam_away):
        if not math.isfinite(lam) or lam <= 0:
            raise ValueError("goal rates must be finite and positive")
    goals = np.arange(max_goals + 1)
    return np.outer(_pois.pmf(goals, lam_home), _pois.pmf(goals, lam_away))


def outcome_probs(m: np.ndarray) -> np.ndarray:
    """[P(home win), P(draw), P(away win)]."""
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def _split(dist: np.ndarray, line: float) -> OverUnder:
    if line < 0:
        raise ValueError("a total line cannot be negative")
    k = np.arange(dist.size)
    return OverUnder(
        over=float(dist[k > line].sum()),
        under=float(dist[k < line].sum()),
        push=float(dist[k == line].sum()),
    )


def over_under(m: np.ndarray, line: float) -> OverUnder:
    """Match total goals against a line, e.g. 2.5."""
    h, a = np.indices(m.shape)
    totals = np.bincount((h + a).ravel(), weights=m.ravel())
    return _split(totals, line)


def team_total(m: np.ndarray, side: str, line: float) -> OverUnder:
    """One team's total goals ("team total points") against a line."""
    if side not in ("home", "away"):
        raise ValueError("side must be 'home' or 'away'")
    return _split(m.sum(axis=1) if side == "home" else m.sum(axis=0), line)


def expected_goals(m: np.ndarray) -> np.ndarray:
    goals = np.arange(m.shape[0])
    return np.array([goals @ m.sum(axis=1), goals @ m.sum(axis=0)])


def expected_total(m: np.ndarray) -> float:
    return float(expected_goals(m).sum())


def fair_odds(p: float) -> float:
    """Decimal odds with no margin for an outcome of probability p."""
    if not 0 < p <= 1:
        raise ValueError("probability must be in (0, 1]")
    return 1.0 / p


def expected_value(p: float, odds: float) -> float:
    """Expected profit per unit staked on an outcome of probability p at decimal odds."""
    return p * odds - 1.0
