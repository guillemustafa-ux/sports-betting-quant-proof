"""Poisson match model. Expected values are computed here with math.exp / math.factorial, not with the
module under test; the total-goals checks use the fact that a sum of independent Poissons is Poisson."""

import math

import numpy as np
import pytest

from sbq import poisson

LH, LA = 1.5, 1.1  # home and away goal rates
LT = LH + LA


def pois(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


@pytest.fixture
def m():
    return poisson.score_matrix(LH, LA, max_goals=20)


def test_score_matrix_cells_are_products_of_two_poissons(m):
    assert m.shape == (21, 21)
    assert m[0, 0] == pytest.approx(math.exp(-LT), rel=1e-12)
    assert m[1, 0] == pytest.approx(LH * math.exp(-LT), rel=1e-12)
    assert m[2, 3] == pytest.approx(pois(2, LH) * pois(3, LA), rel=1e-12)
    assert 1 - m.sum() < 1e-12  # truncation at 20 goals loses nothing measurable


def test_win_draw_loss_probabilities(m):
    home = sum(pois(h, LH) * pois(a, LA) for h in range(21) for a in range(h))
    draw = sum(pois(k, LH) * pois(k, LA) for k in range(21))
    away = sum(pois(h, LH) * pois(a, LA) for a in range(21) for h in range(a))
    np.testing.assert_allclose(poisson.outcome_probs(m), [home, draw, away], rtol=1e-12)
    assert poisson.outcome_probs(m).sum() == pytest.approx(1.0, abs=1e-12)


def test_over_under_on_a_half_line(m):
    under = pois(0, LT) + pois(1, LT) + pois(2, LT)
    ou = poisson.over_under(m, 2.5)
    assert ou.under == pytest.approx(under, rel=1e-12)
    assert ou.over == pytest.approx(1 - under, rel=1e-12)
    assert ou.push == 0.0


def test_over_under_on_a_whole_line_pushes_on_the_line(m):
    ou = poisson.over_under(m, 2.0)
    assert ou.push == pytest.approx(pois(2, LT), rel=1e-12)
    assert ou.under == pytest.approx(pois(0, LT) + pois(1, LT), rel=1e-12)
    assert ou.over + ou.under + ou.push == pytest.approx(1.0, abs=1e-12)


def test_team_total_points(m):
    home = poisson.team_total(m, "home", 1.5)
    assert home.over == pytest.approx(1 - pois(0, LH) - pois(1, LH), rel=1e-12)
    away = poisson.team_total(m, "away", 0.5)
    assert away.under == pytest.approx(pois(0, LA), rel=1e-12)


def test_expected_goals_and_total(m):
    np.testing.assert_allclose(poisson.expected_goals(m), [LH, LA], rtol=1e-10)
    assert poisson.expected_total(m) == pytest.approx(LT, rel=1e-10)


def test_fair_odds_and_expected_value_of_a_bet():
    assert poisson.fair_odds(0.5) == pytest.approx(2.0)
    assert poisson.fair_odds(0.25) == pytest.approx(4.0)
    # a bet that wins 55% of the time at 2.00 returns 0.55 * 2 - 1 = +10% per unit staked
    assert poisson.expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert poisson.expected_value(0.45, 2.0) == pytest.approx(-0.10)


@pytest.mark.parametrize("lh,la", [(0.0, 1.0), (1.0, -0.5), (float("nan"), 1.0)])
def test_rates_must_be_positive(lh, la):
    with pytest.raises(ValueError):
        poisson.score_matrix(lh, la)


def test_bad_line_side_and_probability_are_rejected(m):
    with pytest.raises(ValueError):
        poisson.over_under(m, -0.5)
    with pytest.raises(ValueError):
        poisson.team_total(m, "draw", 1.5)
    with pytest.raises(ValueError):
        poisson.fair_odds(0.0)
