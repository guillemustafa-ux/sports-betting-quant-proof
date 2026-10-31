"""Walk-forward backtest, the two planted leaks, and the timestamp guard that catches them."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from sbq import backtest as bt
from sbq import data

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "data" / "synthetic_league.csv"
FLAT = 1e9  # half-life in days long enough that every past match weighs the same


@pytest.fixture(scope="module")
def matches():
    return data.load_matches(FIXTURE)


@pytest.fixture(scope="module")
def runs(matches):
    return {mode: bt.run(matches, mode) for mode in bt.MODES}


def _history():
    t0 = pd.Timestamp("2023-08-12 15:00")
    return pd.DataFrame({
        "home": ["A", "B"], "away": ["B", "A"], "hg": [2, 1], "ag": [1, 1],
        "result_time": [t0, t0 + pd.Timedelta(days=7)],
    })


def test_rates_match_a_hand_computed_attack_defence_model():
    # league home avg 1.5, away avg 1.0. A scored 3 where 2.5 was expected -> attack 1.2; conceded 2 -> defence 0.8.
    # B: attack 0.8, defence 1.2. A at home to B: 1.5 * 1.2 * 1.2 = 2.16 and 1.0 * 0.8 * 0.8 = 0.64.
    lh, la = bt.rates(_history(), "A", "B", pd.Timestamp("2023-09-01"), half_life_days=FLAT, prior_matches=0)
    assert (lh, la) == (pytest.approx(2.16), pytest.approx(0.64))


def test_rates_shrink_toward_average_with_prior_matches():
    # one pseudo-match of average goals (1.25) on each side: A attack (3 + 1.25) / (2.5 + 1.25)
    lh, _ = bt.rates(_history(), "A", "B", pd.Timestamp("2023-09-01"), half_life_days=FLAT, prior_matches=1)
    att_a, def_b = 4.25 / 3.75, 4.25 / 3.75
    assert lh == pytest.approx(1.5 * att_a * def_b)


def test_guard_rejects_any_feature_at_or_after_kickoff():
    kickoff = pd.Timestamp("2024-01-06 15:00")
    ok = pd.DataFrame({"kickoff": [kickoff], "decision_time": [kickoff - pd.Timedelta(days=1)],
                       "feature_time": [kickoff - pd.Timedelta(days=2)]})
    bt.check_no_leak(ok)
    for bad in (ok.assign(feature_time=kickoff), ok.assign(feature_time=kickoff - pd.Timedelta(hours=1)),
                ok.assign(decision_time=kickoff)):
        with pytest.raises(bt.LeakError):
            bt.check_no_leak(bad)


def test_honest_run_uses_only_data_from_before_each_kickoff(runs):
    pred = runs["honest"].predictions
    assert len(pred) > 900
    assert (pred.feature_time < pred.kickoff).all()
    assert (pred.feature_time < pred.decision_time).all()
    bt.check_no_leak(pred)
    assert runs["honest"].summary["guard"] == "pass"


@pytest.mark.parametrize("mode", ["closing_leak", "random_cv"])
def test_guard_catches_both_planted_leaks(runs, mode):
    with pytest.raises(bt.LeakError):
        bt.check_no_leak(runs[mode].predictions)
    assert runs[mode].summary["guard"] == "FAIL"


def test_every_run_is_scored_on_the_same_matches(runs):
    keys = [tuple(r.predictions.match_id) for r in runs.values()]
    assert keys[0] == keys[1] == keys[2]


def test_honest_predictions_ignore_the_future_and_random_cv_does_not(matches):
    matches = matches.iloc[:700]  # enough history, half the run time
    target = 600
    cut = matches.open_time[target]
    future = matches.kickoff >= cut
    scrambled = matches.copy()
    scrambled.loc[future, ["hg", "ag"]] = scrambled.loc[future, ["ag", "hg"]].to_numpy()
    cols = ["p_h", "p_d", "p_a"]

    def one(df, mode):
        p = bt.run(df, mode).predictions.set_index("match_id")
        return p.loc[target, cols].to_numpy(dtype=float)

    np.testing.assert_allclose(one(matches, "honest"), one(scrambled, "honest"), atol=0)
    assert not np.allclose(one(matches, "random_cv"), one(scrambled, "random_cv"))


def test_bets_settle_at_the_price_taken_and_carry_their_clv():
    pred = pd.DataFrame({
        "match_id": [0, 1, 2], "result": [0, 2, 1],
        "p_h": [0.55, 0.30, 0.34], "p_d": [0.25, 0.25, 0.33], "p_a": [0.20, 0.45, 0.33],
        "open_h": [2.00, 3.00, 2.90], "open_d": [3.60, 3.40, 3.00], "open_a": [4.50, 2.50, 2.90],
        "close_h": [2.00, 3.00, 2.90], "close_d": [3.50, 3.40, 3.00], "close_a": [4.00, 2.40, 2.90],
    })
    bets = bt.place_bets(pred, edge=0.03)
    # match 0: home EV 0.55 * 2.00 - 1 = 0.10 > 0.03 -> bet home, won -> +1.00
    # match 1: away EV 0.45 * 2.50 - 1 = 0.125 -> bet away, won -> +1.50
    # match 2: best EV 0.34 * 2.90 - 1 = -0.014 -> no bet
    assert bets.match_id.tolist() == [0, 1]
    assert bets.selection.tolist() == [0, 2]
    assert bets.profit.tolist() == pytest.approx([1.00, 1.50])
    # CLV of match 0: closing 2.00/3.50/4.00 fair home 14/29 -> 2.00 * 14/29 - 1
    assert bets.clv.iloc[0] == pytest.approx(2.00 * 14 / 29 - 1)


def test_closing_price_leak_inflates_expected_roi_and_clv(runs):
    # Realized ROI on ~1000 bets has a standard error near 6 points, so the assertion reads the two
    # noise-free measures: ROI under the true probabilities and CLV against the close.
    honest, leak = runs["honest"].summary, runs["closing_leak"].summary
    assert honest["true_roi"] < 0 and honest["clv_mean"] < 0
    assert leak["true_roi"] > honest["true_roi"] + 0.03
    assert leak["clv_mean"] > honest["clv_mean"] + 0.03
    assert leak["clv_positive"] > 0.5 > honest["clv_positive"]
    assert leak["close_weight"] > 0.5  # the stacker learned to lean on the leaked column
    assert honest["close_weight"] is None


def test_the_stacker_weight_goes_to_whichever_input_forecasts_better():
    base = {"result": [0, 1, 2], "p_h": [1 / 3] * 3, "p_d": [1 / 3] * 3, "p_a": [1 / 3] * 3}
    right = pd.DataFrame({**base, "fair_h": [0.9, 0.05, 0.05], "fair_d": [0.05, 0.9, 0.05], "fair_a": [0.05, 0.05, 0.9]})
    wrong = pd.DataFrame({**base, "fair_h": [0.05, 0.9, 0.9], "fair_d": [0.9, 0.05, 0.05], "fair_a": [0.05, 0.05, 0.05]})
    assert bt.fit_close_weight(right) == 1.0
    assert bt.fit_close_weight(wrong) == 0.0


def test_expected_roi_reads_the_true_probability_of_each_selection():
    pred = pd.DataFrame({"match_id": [0, 1], "true_h": [0.5, 0.2], "true_d": [0.3, 0.3], "true_a": [0.2, 0.5]})
    bets = pd.DataFrame({"match_id": [0, 1], "selection": [0, 2], "odds": [2.2, 1.8]})
    # (0.5 * 2.2 - 1 + 0.5 * 1.8 - 1) / 2 = (0.10 - 0.10) / 2 = 0
    assert bt.expected_roi(pred, bets) == pytest.approx(0.0)
    assert bt.expected_roi(pred.drop(columns=["true_h"]), bets) is None


def test_random_validation_flatters_the_scores(runs):
    honest, shuffled = runs["honest"].summary, runs["random_cv"].summary
    assert shuffled["brier"] < honest["brier"]
    assert shuffled["log_loss"] < honest["log_loss"]


def test_unknown_mode_is_rejected(matches):
    with pytest.raises(ValueError, match="mode"):
        bt.run(matches, "peeking")
