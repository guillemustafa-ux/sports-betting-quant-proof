"""Closing line value: the price taken against the de-vigged closing probability."""

from fractions import Fraction

import pytest

from sbq import clv


def test_clv_of_a_bet_that_beat_the_close():
    # closing book 2.00 / 3.50 / 4.00 -> multiplicative fair home probability 14/29 (see test_devig)
    # home taken at 2.20 -> 2.20 * 14/29 - 1 = 1.8/29
    value = clv.clv(2.20, [2.00, 3.50, 4.00], selection=0)
    assert value == pytest.approx(float(Fraction(18, 290)), abs=1e-12)


def test_clv_is_negative_when_the_price_taken_was_worse_than_the_fair_close():
    # symmetric close 1.90 / 1.90 -> fair 0.5 each; taking 1.90 is 1.90 * 0.5 - 1 = -5%
    assert clv.clv(1.90, [1.90, 1.90], selection=1) == pytest.approx(-0.05)


def test_clv_uses_the_requested_devig_method():
    close = [1.25, 6.0, 11.0]
    by_mult = clv.clv(12.0, close, selection=2, method="multiplicative")
    by_shin = clv.clv(12.0, close, selection=2, method="shin")
    assert by_shin < by_mult  # Shin gives the longshot less fair probability


def test_selection_must_exist_in_the_closing_market():
    with pytest.raises(ValueError):
        clv.clv(2.0, [1.9, 1.9], selection=2)


def test_summary_aggregates_per_bet_clv():
    s = clv.summarize([0.10, -0.05, 0.02])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(0.07 / 3)
    assert s["median"] == pytest.approx(0.02)
    assert s["share_positive"] == pytest.approx(2 / 3)


def test_summary_of_no_bets_is_explicit():
    assert clv.summarize([]) == {"n": 0, "mean": None, "median": None, "share_positive": None}
