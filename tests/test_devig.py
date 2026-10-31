"""Vig removal: every expected number here is computed by hand (exact fractions) or by running a
method's own forward model with known inputs and checking that the method inverts it."""

from fractions import Fraction

import numpy as np
import pytest

from sbq import devig

METHODS = ["multiplicative", "shin", "power"]


def test_overround_of_a_symmetric_two_way_book():
    # 1/1.90 + 1/1.90 = 1.052631..., so the book carries a 5.26% margin
    assert devig.overround([1.90, 1.90]) == pytest.approx(2 / 1.90 - 1, abs=1e-12)
    assert devig.overround([2.0, 2.0]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_a_book_without_margin_is_returned_as_is(method):
    np.testing.assert_allclose(devig.devig([2.0, 4.0, 4.0], method), [0.5, 0.25, 0.25], atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_a_symmetric_two_way_book_is_fifty_fifty(method):
    np.testing.assert_allclose(devig.devig([1.90, 1.90], method), [0.5, 0.5], atol=1e-12)


def test_multiplicative_three_way_matches_exact_fractions():
    # implied 1/2, 2/7, 1/4 sum to 29/28; dividing by it gives 14/29, 8/29, 7/29
    expected = [float(Fraction(14, 29)), float(Fraction(8, 29)), float(Fraction(7, 29))]
    np.testing.assert_allclose(devig.multiplicative([2.0, 3.5, 4.0]), expected, atol=1e-12)


def test_shin_on_two_outcomes_equals_splitting_the_margin_equally():
    # Known result: with two outcomes Shin's method subtracts half the overround from each side.
    # 1/1.8 = 5/9, 1/2.1 = 10/21 -> home = (5/9 - 10/21 + 1) / 2 = 34/63
    home = float(Fraction(34, 63))
    np.testing.assert_allclose(devig.shin([1.8, 2.1]), [home, 1 - home], atol=1e-9)


def test_shin_inverts_its_own_forward_model():
    # Shin (1993): with true probabilities p and insider share z the book prices
    # pi_i = sqrt(z p_i + (1 - z) p_i^2) * sum_j sqrt(z p_j + (1 - z) p_j^2)
    p, z = np.array([0.5, 0.3, 0.2]), 0.03
    root = np.sqrt(z * p + (1 - z) * p**2)
    odds = 1 / (root * root.sum())
    np.testing.assert_allclose(devig.shin(odds), p, atol=1e-9)
    assert devig.shin_z(odds) == pytest.approx(z, abs=1e-9)


def test_power_inverts_its_own_forward_model():
    # Power method: fair p_i = pi_i ** k. Build pi_i = p_i ** (1 / k) with k = 1.08.
    p, k = np.array([0.6, 0.25, 0.15]), 1.08
    odds = 1 / p ** (1 / k)
    assert devig.overround(odds) > 0
    np.testing.assert_allclose(devig.power(odds), p, atol=1e-9)
    assert devig.power_k(odds) == pytest.approx(k, abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_fair_probabilities_sum_to_one_on_a_real_looking_1x2_book(method):
    fair = devig.devig([2.10, 3.40, 3.60], method)
    assert fair.sum() == pytest.approx(1.0, abs=1e-9)
    assert (fair > 0).all() and (fair < 1).all()


@pytest.mark.parametrize("method", ["shin", "power"])
def test_shin_and_power_load_the_margin_on_the_longshot(method):
    # The favourite-longshot bias correction: compared with the proportional method, these two give
    # the favourite more probability and the longshot less.
    odds = [1.25, 6.0, 11.0]
    mult, other = devig.multiplicative(odds), devig.devig(odds, method)
    assert other[0] > mult[0]
    assert other[2] < mult[2]


@pytest.mark.parametrize("bad", [[1.0, 3.0], [0.5, 3.0], [2.0], []])
@pytest.mark.parametrize("method", METHODS)
def test_invalid_books_are_rejected(method, bad):
    with pytest.raises(ValueError):
        devig.devig(bad, method)


def test_shin_rejects_a_book_with_negative_margin():
    with pytest.raises(ValueError, match="negative overround"):
        devig.shin([2.2, 2.2])


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        devig.devig([1.9, 1.9], "additive")
