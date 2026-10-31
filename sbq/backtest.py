"""Walk-forward backtest of a Poisson rating model, with two leaks that can be switched on.

Modes:
- honest: before each match, rate every team from the results known at the moment the price is taken
  (open_time), and nothing else.
- closing_leak: the same ratings, plus the de-vigged closing price as a second feature, the way a
  closing-odds column slips into a feature table. A one-weight stacker blends the two, fitted by log
  loss on the whole sample, as a model trained with that column would; it learns to lean on the close.
  The closing line only exists at kickoff.
- random_cv: ratings fitted on shuffled K-fold training sets, so matches played after the one being
  predicted inform its rating ("random instead of chronological validation").

Every prediction carries `decision_time` (when the bet is placed) and `feature_time` (the latest
timestamp of any data it used). `check_no_leak` refuses any row whose features are not strictly older
than the decision, or whose decision is not before kickoff.

Bets: flat one unit on the outcome with the highest expected value at the pre-closing price, when that
value exceeds `edge`. Settled at the price taken; CLV is measured against the de-vigged close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sbq import calibration, clv, poisson
from sbq.devig import devig

MODES = ("honest", "closing_leak", "random_cv")
OUTCOMES = ("h", "d", "a")


class LeakError(AssertionError):
    """A prediction used data that did not exist when the bet was placed."""


@dataclass(frozen=True)
class Config:
    half_life_days: float = 120.0  # weight of a past match halves every 120 days
    prior_matches: float = 4.0  # pseudo-matches of league-average goals mixed into every rating
    min_team_matches: int = 6  # both teams need this many known results before a match is scored
    edge: float = 0.03  # minimum expected value at the price taken to place a bet
    folds: int = 5
    seed: int = 7
    method: str = "multiplicative"  # de-vig method for the closing line


@dataclass
class Run:
    mode: str
    predictions: pd.DataFrame
    bets: pd.DataFrame
    summary: dict = field(default_factory=dict)


def rates(history: pd.DataFrame, home: str, away: str, at: pd.Timestamp,
          half_life_days: float, prior_matches: float) -> tuple[float, float]:
    """Expected goals of `home` and `away` from an attack/defence model fitted on `history`.

    attack = goals scored / goals a league-average side would score in the same fixtures,
    defence = goals conceded / goals expected against, each shrunk toward 1 by `prior_matches`
    pseudo-matches; past matches weigh 0.5 ** (days away / half_life_days).
    """
    age = np.abs((at - history.result_time).dt.total_seconds().to_numpy()) / 86400.0
    w = 0.5 ** (age / half_life_days)
    hg, ag = history.hg.to_numpy(float), history.ag.to_numpy(float)
    home_avg, away_avg = (w * hg).sum() / w.sum(), (w * ag).sum() / w.sum()
    prior = prior_matches * (home_avg + away_avg) / 2

    def strength(team: str) -> tuple[float, float]:
        at_home = w * (history.home == team).to_numpy()
        away_ = w * (history.away == team).to_numpy()
        scored = (at_home * hg).sum() + (away_ * ag).sum()
        conceded = (at_home * ag).sum() + (away_ * hg).sum()
        exp_scored = at_home.sum() * home_avg + away_.sum() * away_avg
        exp_conceded = at_home.sum() * away_avg + away_.sum() * home_avg
        return (scored + prior) / (exp_scored + prior), (conceded + prior) / (exp_conceded + prior)

    att_h, def_h = strength(home)
    att_a, def_a = strength(away)
    return home_avg * att_h * def_a, away_avg * att_a * def_h


def _fair(row: pd.Series, prefix: str, method: str) -> np.ndarray:
    return devig([row[f"{prefix}_{o}"] for o in OUTCOMES], method)


def predict(matches: pd.DataFrame, mode: str, cfg: Config = Config()) -> pd.DataFrame:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; use one of {MODES}")
    fold = np.random.default_rng(cfg.seed).permutation(len(matches)) % cfg.folds
    rows = []
    for i, m in matches.iterrows():
        known = matches[matches.result_time < m.open_time]
        counts = pd.concat([known.home, known.away]).value_counts()
        if min(counts.get(m.home, 0), counts.get(m.away, 0)) < cfg.min_team_matches:
            continue  # the same matches are skipped in every mode, so all runs score the same set
        train = matches[fold != fold[i]] if mode == "random_cv" else known
        lh, la = rates(train, m.home, m.away, m.open_time, cfg.half_life_days, cfg.prior_matches)
        grid = poisson.score_matrix(lh, la, max_goals=12)
        grid /= grid.sum()
        p = poisson.outcome_probs(grid)
        p_over = poisson.over_under(grid, 2.5).over
        feature_time = train.result_time.max()
        if mode == "closing_leak":
            feature_time = max(feature_time, m.close_time)
        close = _fair(m, "close", cfg.method)
        close_over = devig([m.close_over, m.close_under], cfg.method)[0] if pd.notna(m.close_over) else np.nan
        rows.append({
            "match_id": i, "kickoff": m.kickoff, "decision_time": m.open_time, "feature_time": feature_time,
            "home": m.home, "away": m.away, "result": m.result, "over": int(m.hg + m.ag > 2.5),
            "lam_h": lh, "lam_a": la, "p_h": p[0], "p_d": p[1], "p_a": p[2], "p_over": p_over,
            **{f"{s}_{o}": m[f"{s}_{o}"] for s in ("open", "close") for o in OUTCOMES},
            "close_over": m.close_over, "close_under": m.close_under,
            "fair_h": close[0], "fair_d": close[1], "fair_a": close[2], "fair_over": close_over,
            **{c: m[c] for c in ("true_h", "true_d", "true_a", "true_over") if c in m},
        })
    pred = pd.DataFrame(rows)
    pred.attrs["close_weight"] = None
    if mode == "closing_leak":
        w = fit_close_weight(pred)
        model, fair = pred[["p_h", "p_d", "p_a"]].to_numpy(), pred[["fair_h", "fair_d", "fair_a"]].to_numpy()
        pred[["p_h", "p_d", "p_a"]] = (1 - w) * model + w * fair
        pred["p_over"] = np.where(pred.fair_over.notna(), (1 - w) * pred.p_over + w * pred.fair_over, pred.p_over)
        pred.attrs["close_weight"] = w
    return pred


def fit_close_weight(pred: pd.DataFrame) -> float:
    """Weight w in [0, 1] of the closing probability that minimises the log loss of
    (1 - w) * model + w * close over every match: the leaky stacker."""
    model, fair = pred[["p_h", "p_d", "p_a"]].to_numpy(), pred[["fair_h", "fair_d", "fair_a"]].to_numpy()
    grid = np.linspace(0, 1, 101)
    losses = [calibration.log_loss((1 - w) * model + w * fair, pred.result) for w in grid]
    return float(grid[int(np.argmin(losses))])


def check_no_leak(pred: pd.DataFrame) -> None:
    late = pred.feature_time >= pred.decision_time
    after = pred.decision_time >= pred.kickoff
    if late.any() or after.any():
        raise LeakError(f"{int(late.sum())} predictions use data from at or after the bet; "
                        f"{int(after.sum())} bets are placed at or after kickoff")


def place_bets(pred: pd.DataFrame, edge: float = Config.edge, method: str = Config.method) -> pd.DataFrame:
    rows = []
    for _, r in pred.iterrows():
        p = np.array([r.p_h, r.p_d, r.p_a])
        odds = np.array([r.open_h, r.open_d, r.open_a])
        ev = p * odds - 1
        pick = int(ev.argmax())
        if ev[pick] <= edge:
            continue
        won = pick == r.result
        close = [r.close_h, r.close_d, r.close_a]
        rows.append({"match_id": r.match_id, "selection": pick, "odds": odds[pick], "ev": ev[pick],
                     "won": won, "profit": odds[pick] - 1 if won else -1.0,
                     "clv": clv.clv(odds[pick], close, pick, method)})
    return pd.DataFrame(rows, columns=["match_id", "selection", "odds", "ev", "won", "profit", "clv"])


def run(matches: pd.DataFrame, mode: str, cfg: Config = Config()) -> Run:
    pred = predict(matches, mode, cfg)
    bets = place_bets(pred, cfg.edge, cfg.method)
    probs = pred[["p_h", "p_d", "p_a"]].to_numpy()
    try:
        check_no_leak(pred)
        guard = "pass"
    except LeakError:
        guard = "FAIL"
    c = clv.summarize(bets.clv)
    n = len(bets)
    summary = {
        "mode": mode, "guard": guard, "matches": len(pred), "bets": n,
        "roi": bets.profit.mean() if n else None,
        "roi_se": bets.profit.std(ddof=1) / np.sqrt(n) if n > 1 else None,
        "true_roi": expected_roi(pred, bets),
        "clv_mean": c["mean"], "clv_positive": c["share_positive"],
        "brier": calibration.brier(probs, pred.result), "log_loss": calibration.log_loss(probs, pred.result),
        "close_weight": pred.attrs["close_weight"],
    }
    return Run(mode, pred, bets, summary)


def expected_roi(pred: pd.DataFrame, bets: pd.DataFrame) -> float | None:
    """Mean of true_p * odds - 1 over the bets: the ROI without result noise. Synthetic data only."""
    if "true_h" not in pred or bets.empty:
        return None
    truth = pred.set_index("match_id").loc[bets.match_id, ["true_h", "true_d", "true_a"]].to_numpy()
    truth = truth / truth.sum(axis=1, keepdims=True)  # the CSV rounds them to 4 decimals
    p = truth[np.arange(len(bets)), bets.selection.to_numpy(int)]
    return float((p * bets.odds.to_numpy() - 1).mean())
