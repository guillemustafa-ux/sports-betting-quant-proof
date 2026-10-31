"""Render the audit as Markdown: the backtest table, calibration against the market, reliability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sbq import calibration
from sbq.backtest import Run
from sbq.devig import devig, overround

LABELS = {
    "honest": ("Honest walk-forward", "chronological, data before the bet only"),
    "closing_leak": ("Leak: closing price in features", "chronological, closing odds as a feature"),
    "random_cv": ("Leak: random K-fold", "random 5-fold instead of chronological"),
}
SIDES = ["h", "d", "a"]


def _pct(x, signed=True) -> str:
    return "n/a" if x is None else (f"{x:+.1%}" if signed else f"{x:.0%}")


def _audit_table(runs: dict[str, Run]) -> list[str]:
    lines = [
        "| Run | Validation | Leak guard | Bets | Realized ROI +/- SE | Expected ROI under the truth | Mean CLV | Bets with CLV > 0 | Brier | Log loss |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, r in runs.items():
        s = r.summary
        roi = "n/a" if s["roi"] is None else f"{s['roi']:+.1%} +/- {s['roi_se']:.1%}"
        name, how = LABELS[mode]
        lines.append(f"| {name} | {how} | {s['guard']} | {s['bets']} | {roi} | {_pct(s['true_roi'])} | "
                     f"{_pct(s['clv_mean'])} | {_pct(s['clv_positive'], signed=False)} | {s['brier']:.4f} | {s['log_loss']:.4f} |")
    return lines


def _market(pred: pd.DataFrame, prefix: str, method: str) -> np.ndarray:
    return np.array([devig(row, method) for row in pred[[f"{prefix}_{s}" for s in SIDES]].to_numpy()])


def _calibration_table(pred: pd.DataFrame) -> tuple[list[str], dict[str, np.ndarray]]:
    forecasters = {"Model (honest walk-forward)": pred[["p_h", "p_d", "p_a"]].to_numpy()}
    forecasters["Opening market, multiplicative de-vig"] = _market(pred, "open", "multiplicative")
    for method in ("multiplicative", "shin", "power"):
        forecasters[f"Closing market, {method} de-vig"] = _market(pred, "close", method)
    if "true_h" in pred:
        truth = pred[["true_h", "true_d", "true_a"]].to_numpy()
        forecasters["Truth (synthetic only)"] = truth / truth.sum(axis=1, keepdims=True)
    card = calibration.scorecard(forecasters, pred.result)
    lines = ["| Forecaster | Brier | Log loss |", "|---|---|---|"]
    lines += [f"| {name} | {row.brier:.4f} | {row.log_loss:.4f} |" for name, row in card.iterrows()]
    return lines, forecasters


def _over_under_line(pred: pd.DataFrame) -> str:
    ou = pred[pred.fair_over.notna()]
    model = np.column_stack([ou.p_over, 1 - ou.p_over])
    close = np.column_stack([ou.fair_over, 1 - ou.fair_over])
    y = 1 - ou.over.to_numpy()  # index 0 = over happened
    return (f"Over/under 2.5 goals on {len(ou)} matches: model Brier {calibration.brier(model, y):.4f}, "
            f"closing market Brier {calibration.brier(close, y):.4f}.")


def _reliability(model: np.ndarray, close: np.ndarray, outcome) -> list[str]:
    a = calibration.reliability_table(model, outcome).set_index("bin_lo")
    b = calibration.reliability_table(close, outcome).set_index("bin_lo")
    lines = ["| Forecast bin | Model: n | Model: mean forecast | Model: observed | Close: n | Close: mean forecast | Close: observed |",
             "|---|---|---|---|---|---|---|"]
    for lo in sorted(set(a.index) | set(b.index)):
        cells = []
        for t in (a, b):
            cells += ([str(int(t.loc[lo, "n"])), f"{t.loc[lo, 'mean_pred']:.3f}", f"{t.loc[lo, 'observed']:.3f}"]
                      if lo in t.index else ["0", "", ""])
        lines.append(f"| {lo:.1f}-{lo + 0.1:.1f} | " + " | ".join(cells) + " |")
    return lines


def render(matches: pd.DataFrame, runs: dict[str, Run], source: str) -> str:
    honest = runs["honest"].predictions
    cfg = runs["honest"].config
    cal_lines, forecasters = _calibration_table(honest)
    margin_open = np.mean([overround(r) for r in honest[[f"open_{s}" for s in SIDES]].to_numpy()])
    margin_close = np.mean([overround(r) for r in honest[[f"close_{s}" for s in SIDES]].to_numpy()])
    out = [
        "# Results",
        "",
        f"Data: `{source}`, {len(matches)} matches, {len(honest)} scored (both teams need "
        f"{cfg.min_team_matches} known results first; every run is scored on the same matches).",
        f"Bets: one unit at the pre-closing price on the outcome with the highest expected value, when it exceeds "
        f"{cfg.edge:.0%}. CLV against the closing line, {cfg.method} de-vig. Mean overround: opening "
        f"{margin_open:.2%}, closing {margin_close:.2%}.",
        "",
        "## Backtest audit",
        "",
        *_audit_table(runs),
        "",
        f"The closing-price stacker put weight {runs['closing_leak'].summary['close_weight']:.2f} on the leaked column.",
        "",
        "## Calibration against the market (1X2, honest model's matches)",
        "",
        *cal_lines,
        "",
        _over_under_line(honest),
        "",
        "## Reliability: honest model vs. closing market (every outcome forecast, 10 bins)",
        "",
        *_reliability(forecasters["Model (honest walk-forward)"],
                      forecasters["Closing market, multiplicative de-vig"], honest.result),
        "",
    ]
    return "\n".join(out)
