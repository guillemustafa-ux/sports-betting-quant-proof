"""A synthetic league where the truth is known, written in football-data.co.uk's CSV layout.

Twenty teams play three double round-robin seasons. Each team has an attack and a defence strength
that drift every round, so recent results matter more than old ones. Goals are Poisson from those
strengths. Two books price every match:

- the opening ("pre-closing") book: the true probabilities with noise of sd NOISE_OPEN on the log
  scale, then a MARGIN_OPEN margin;
- the closing book: keeps CLOSE_KEEPS of the opening error plus fresh noise of sd NOISE_CLOSE, then
  a MARGIN_CLOSE margin. The market moves toward the truth between open and close, as real markets do.

The opening book is built sharper than a results-only rating model can be, so an honest model is
not expected to find an edge; the true probabilities are written next to the prices (TrueH, TrueD,
TrueA, TrueO25) so every forecast can also be scored against the truth.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from sbq import poisson

SEED = 20261031
N_TEAMS = 20
SEASONS = 3
FIRST_KICKOFF = pd.Timestamp("2021-08-14 15:00")  # a Saturday; one round a week
BASE_HOME, BASE_AWAY = 1.50, 1.15
STRENGTH_SD, DRIFT_SD = 0.25, 0.02
NOISE_OPEN, NOISE_CLOSE, CLOSE_KEEPS = 0.08, 0.03, 0.4
MARGIN_OPEN, MARGIN_CLOSE = 0.03, 0.025
LINE = 2.5


def _round_robin(n: int) -> list[list[tuple[int, int]]]:
    """Circle method: n - 1 rounds, then the same rounds with venues swapped."""
    teams = list(range(n))
    rounds = []
    for r in range(n - 1):
        pairs = [(teams[i], teams[n - 1 - i]) for i in range(n // 2)]
        rounds.append([(a, b) if (r + i) % 2 == 0 else (b, a) for i, (a, b) in enumerate(pairs)])
        teams = [teams[0], teams[-1], *teams[1:-1]]
    return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]


def _book(log_p: np.ndarray, margin: float) -> np.ndarray:
    q = np.exp(log_p - log_p.max())
    q /= q.sum()
    return np.maximum(np.round(1 / (q * (1 + margin)), 2), 1.01)


def league(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    attack = rng.normal(0, STRENGTH_SD, N_TEAMS)
    defence = rng.normal(0, STRENGTH_SD, N_TEAMS)  # higher = concedes more
    names = [f"Team {i + 1:02d}" for i in range(N_TEAMS)]
    rows = []
    week = 0
    for _ in range(SEASONS):
        for rnd in _round_robin(N_TEAMS):
            kickoff = FIRST_KICKOFF + pd.Timedelta(weeks=week)
            for h, a in rnd:
                lh = BASE_HOME * np.exp(attack[h] + defence[a])
                la = BASE_AWAY * np.exp(attack[a] + defence[h])
                m = poisson.score_matrix(lh, la, max_goals=12)
                m /= m.sum()
                p1x2 = poisson.outcome_probs(m)
                ou = poisson.over_under(m, LINE)
                p_ou = np.array([ou.over, ou.under])
                hg, ag = rng.poisson(lh), rng.poisson(la)
                e1, e2 = rng.normal(0, NOISE_OPEN, 3), rng.normal(0, NOISE_OPEN, 2)
                c1 = CLOSE_KEEPS * e1 + rng.normal(0, NOISE_CLOSE, 3)
                c2 = CLOSE_KEEPS * e2 + rng.normal(0, NOISE_CLOSE, 2)
                rows.append({
                    "Div": "SYN", "Date": kickoff.strftime("%d/%m/%Y"), "Time": kickoff.strftime("%H:%M"),
                    "HomeTeam": names[h], "AwayTeam": names[a], "FTHG": hg, "FTAG": ag,
                    "FTR": "H" if hg > ag else ("D" if hg == ag else "A"),
                    **dict(zip(["PSH", "PSD", "PSA"], _book(np.log(p1x2) + e1, MARGIN_OPEN))),
                    **dict(zip(["P>2.5", "P<2.5"], _book(np.log(p_ou) + e2, MARGIN_OPEN))),
                    **dict(zip(["PSCH", "PSCD", "PSCA"], _book(np.log(p1x2) + c1, MARGIN_CLOSE))),
                    **dict(zip(["PC>2.5", "PC<2.5"], _book(np.log(p_ou) + c2, MARGIN_CLOSE))),
                    "TrueH": p1x2[0], "TrueD": p1x2[1], "TrueA": p1x2[2], "TrueO25": ou.over,
                })
            week += 1
            attack += rng.normal(0, DRIFT_SD, N_TEAMS)
            defence += rng.normal(0, DRIFT_SD, N_TEAMS)
            attack -= attack.mean()
            defence -= defence.mean()
    return pd.DataFrame(rows)


def write_league(path: str | pathlib.Path, seed: int = SEED) -> None:
    df = league(seed)
    prices = ["PSH", "PSD", "PSA", "P>2.5", "P<2.5", "PSCH", "PSCD", "PSCA", "PC>2.5", "PC<2.5"]
    df[prices] = df[prices].map(lambda x: f"{x:.2f}")
    df[["TrueH", "TrueD", "TrueA", "TrueO25"]] = df[["TrueH", "TrueD", "TrueA", "TrueO25"]].map(lambda x: f"{x:.4f}")
    df.to_csv(path, index=False, lineterminator="\n")


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parents[1] / "data" / "synthetic_league.csv"
    write_league(out)
    print(f"wrote {out}")
