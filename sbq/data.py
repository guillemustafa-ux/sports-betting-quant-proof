"""Load a football-data.co.uk style CSV into one row per match, with a timestamp on every record.

Prices: Pinnacle (PS*, P*) when present, the market average (Avg*) otherwise, per cell. The pre-closing
columns are the price a bet is taken at ("open"); the C columns are the closing line.

Timestamps (the leak guard reads these):
- open_time: when the pre-closing price is taken. football-data.co.uk collects it on Friday or Tuesday
  afternoon before the round; one day before kickoff is used as a conservative approximation.
- close_time: the closing line exists only at kickoff.
- result_time: a result is known at kickoff + 2 hours.
"""

from __future__ import annotations

import pathlib

import pandas as pd

OPEN_LEAD = pd.Timedelta(days=1)
RESULT_LAG = pd.Timedelta(hours=2)
DEFAULT_TIME = "15:00"  # older seasons carry no Time column

# our column -> source columns in order of preference
PRICES = {
    "open_h": ["PSH", "AvgH"], "open_d": ["PSD", "AvgD"], "open_a": ["PSA", "AvgA"],
    "close_h": ["PSCH", "AvgCH"], "close_d": ["PSCD", "AvgCD"], "close_a": ["PSCA", "AvgCA"],
    "open_over": ["P>2.5", "Avg>2.5"], "open_under": ["P<2.5", "Avg<2.5"],
    "close_over": ["PC>2.5", "AvgC>2.5"], "close_under": ["PC<2.5", "AvgC<2.5"],
}
TRUTH = {"true_h": "TrueH", "true_d": "TrueD", "true_a": "TrueA", "true_over": "TrueO25"}
REQUIRED = ["open_h", "open_d", "open_a", "close_h", "close_d", "close_a"]
RESULT = {"H": 0, "D": 1, "A": 2}


def load_matches(path: str | pathlib.Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    time = raw["Time"].fillna(DEFAULT_TIME) if "Time" in raw else DEFAULT_TIME
    out = pd.DataFrame({
        "kickoff": pd.to_datetime(raw["Date"] + " " + time, dayfirst=True, format="mixed"),
        "home": raw["HomeTeam"],
        "away": raw["AwayTeam"],
        "hg": raw["FTHG"].astype(int),
        "ag": raw["FTAG"].astype(int),
        "result": raw["FTR"].map(RESULT),
    })
    for col, sources in PRICES.items():
        series = pd.Series(float("nan"), index=raw.index)
        for src in sources:
            if src in raw:
                series = series.fillna(raw[src].astype(float))
        out[col] = series
    for col, src in TRUTH.items():  # only the synthetic league knows the true probabilities
        if src in raw:
            out[col] = raw[src].astype(float)
    out = out.dropna(subset=REQUIRED)
    out["open_time"] = out.kickoff - OPEN_LEAD
    out["close_time"] = out.kickoff
    out["result_time"] = out.kickoff + RESULT_LAG
    return out.sort_values("kickoff", kind="stable").reset_index(drop=True)
