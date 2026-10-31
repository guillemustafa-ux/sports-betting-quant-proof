"""Remove the bookmaker's margin from decimal odds.

Three methods, each a different assumption about where the margin sits:

- multiplicative (proportional): the margin is spread in proportion to each implied probability.
- shin: the margin protects the book against a share z of insider money (Shin, 1993); it loads
  more of the margin on longshots, which matches the favourite-longshot bias.
- power: fair p_i = (1 / odds_i) ** k, with k chosen so the probabilities sum to one.

Every function takes the decimal odds of one market (2-way or 3-way, any number of outcomes >= 2)
and returns fair probabilities that sum to one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq

_TOL = 1e-12


def implied(odds: Sequence[float]) -> np.ndarray:
    """Raw implied probabilities 1 / odds, validated."""
    o = np.asarray(odds, dtype=float)
    if o.ndim != 1 or o.size < 2:
        raise ValueError("a market needs at least two outcomes")
    if not np.isfinite(o).all() or (o <= 1.0).any():
        raise ValueError("decimal odds must be finite and greater than 1")
    return 1.0 / o


def overround(odds: Sequence[float]) -> float:
    """Sum of implied probabilities minus one: the book's margin."""
    return float(implied(odds).sum() - 1.0)


def multiplicative(odds: Sequence[float]) -> np.ndarray:
    pi = implied(odds)
    return pi / pi.sum()


def _shin_probs(pi: np.ndarray, z: float) -> np.ndarray:
    s = pi.sum()
    return (np.sqrt(z * z + 4 * (1 - z) * pi**2 / s) - z) / (2 * (1 - z))


def shin_z(odds: Sequence[float]) -> float:
    """Shin's insider share z: the value that makes the fair probabilities sum to one."""
    pi = implied(odds)
    margin = pi.sum() - 1.0
    if margin < -_TOL:
        raise ValueError("negative overround: Shin's model needs a book with a margin")
    if margin <= _TOL:
        return 0.0
    return float(brentq(lambda z: _shin_probs(pi, z).sum() - 1.0, 0.0, 0.999, xtol=1e-14))


def shin(odds: Sequence[float]) -> np.ndarray:
    pi = implied(odds)
    probs = _shin_probs(pi, shin_z(odds))
    return probs / probs.sum()  # removes the root finder's last ~1e-14


def power_k(odds: Sequence[float]) -> float:
    """Exponent k with sum((1 / odds) ** k) == 1; k > 1 when the book has a margin."""
    pi = implied(odds)
    if abs(pi.sum() - 1.0) <= _TOL:
        return 1.0
    return float(brentq(lambda k: (pi**k).sum() - 1.0, 0.01, 100.0, xtol=1e-14))


def power(odds: Sequence[float]) -> np.ndarray:
    probs = implied(odds) ** power_k(odds)
    return probs / probs.sum()


_METHODS = {"multiplicative": multiplicative, "shin": shin, "power": power}


def devig(odds: Sequence[float], method: str = "multiplicative") -> np.ndarray:
    """Fair probabilities of one market by the named method."""
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(f"unknown method {method!r}; use one of {sorted(_METHODS)}") from None
    return fn(odds)
