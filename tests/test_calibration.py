"""Brier, log loss and the reliability table, checked against numbers worked out by hand."""

import math

import pytest

from sbq import calibration as cal


def test_brier_sums_squared_errors_over_outcomes_and_averages_over_matches():
    # row 1: home happened -> 0.5^2 + 0.3^2 + 0.2^2 = 0.38
    # row 2: draw happened -> 0.2^2 + 0.7^2 + 0.5^2 = 0.78; home happened -> 0.8^2 + 0.3^2 + 0.5^2 = 0.98
    probs = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]
    assert cal.brier(probs, [0, 0]) == pytest.approx((0.38 + 0.98) / 2)
    assert cal.brier(probs, [0, 1]) == pytest.approx((0.38 + 0.78) / 2)
    assert cal.brier([[1.0, 0.0]], [0]) == 0.0


def test_log_loss_is_the_mean_negative_log_of_the_probability_given_to_what_happened():
    probs = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]
    assert cal.log_loss(probs, [0, 2]) == pytest.approx(math.log(2))
    assert cal.log_loss(probs, [1, 1]) == pytest.approx(-math.log(0.3))


def test_log_loss_stays_finite_when_the_outcome_was_given_zero():
    value = cal.log_loss([[1.0, 0.0]], [1])
    assert math.isfinite(value) and value == pytest.approx(-math.log(1e-15))


def test_reliability_table_bins_every_outcome_forecast():
    # 4 two-way forecasts -> 8 (probability, happened) pairs:
    # (.9,1) (.1,0) (.85,0) (.15,1) (.2,0) (.8,1) (.15,0) (.85,1)
    probs = [[0.9, 0.1], [0.85, 0.15], [0.2, 0.8], [0.15, 0.85]]
    table = cal.reliability_table(probs, [0, 1, 1, 1], bins=10)
    assert list(table.columns) == ["bin_lo", "bin_hi", "n", "mean_pred", "observed"]
    rows = {round(r.bin_lo, 1): r for r in table.itertuples()}
    assert sorted(rows) == [0.1, 0.2, 0.8, 0.9]
    assert rows[0.1].n == 3 and rows[0.1].mean_pred == pytest.approx(0.4 / 3) and rows[0.1].observed == pytest.approx(1 / 3)
    assert rows[0.2].n == 1 and rows[0.2].observed == 0.0
    assert rows[0.8].n == 3 and rows[0.8].mean_pred == pytest.approx(2.5 / 3) and rows[0.8].observed == pytest.approx(2 / 3)
    assert rows[0.9].n == 1 and rows[0.9].bin_hi == pytest.approx(1.0) and rows[0.9].observed == 1.0
    assert table.n.sum() == 8


def test_a_probability_of_one_lands_in_the_top_bin():
    table = cal.reliability_table([[1.0, 0.0]], [0], bins=5)
    assert table.bin_lo.tolist() == [0.0, pytest.approx(0.8)]
    assert table.n.tolist() == [1, 1]


def test_scorecard_ranks_a_sharper_correct_forecaster_ahead():
    outcome = [0, 2, 1]
    sharp = [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]]
    flat = [[1 / 3] * 3] * 3
    card = cal.scorecard({"sharp": sharp, "flat": flat}, outcome)
    assert card.index.tolist() == ["sharp", "flat"]
    assert card.loc["flat", "brier"] == pytest.approx(2 / 3)  # (2/3)^2 + 2 * (1/3)^2
    assert card.loc["flat", "log_loss"] == pytest.approx(math.log(3))
    assert card.loc["sharp", "brier"] < card.loc["flat", "brier"]
    assert card.loc["sharp", "log_loss"] < card.loc["flat", "log_loss"]
    assert card.n.tolist() == [3, 3]


@pytest.mark.parametrize("probs,outcome", [([[0.5, 0.5]], [2]), ([[0.5, 0.5]], [0, 1]), ([[0.7, 0.7]], [0])])
def test_malformed_inputs_are_rejected(probs, outcome):
    with pytest.raises(ValueError):
        cal.brier(probs, outcome)
