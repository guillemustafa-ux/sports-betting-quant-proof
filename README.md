# sports-betting-quant-proof

Odds math done by hand and tested against hand-checked numbers (margin removal, a Poisson match model, calibration, closing line value), and a betting backtest audited the way a leak-free backtest has to be: a timestamp guard that fails any prediction using data from after the bet, and two planted leaks (*"Closing prices accidentally entering model features"*, *"Random instead of chronological validation"*) measured against the honest run.

The point is not a winning model. There is none here and none is claimed. The point is that a betting backtest's ROI is the noisiest number it reports, and the two most common leaks make it look better than anything you could bet live. On a synthetic league the true probabilities are known, so the audit can say exactly how much of each result is edge and how much is luck or leakage.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[test]"   # Windows; on macOS/Linux use .venv/bin/pip
python -m sbq                 # the audit on the committed synthetic league: writes RESULTS.md in about 12 seconds
python -m pytest              # 84 tests, offline
python data/fetch_football_data.py                                       # optional: one real season, for your private use
python -m sbq --csv data/raw/E0_2324.csv --out data/raw/RESULTS_E0_2324.md   # the same audit on it
```

Same seed, same numbers: `RESULTS.md` regenerates byte-identical, and a test fails if the table below stops matching it.

## The backtest audit

A Poisson attack/defence rating model, refitted before every match. Bets: one unit at the pre-closing price on the outcome with the highest expected value, when it exceeds 3%. 1,140 synthetic matches (three seasons, twenty teams), 1,080 scored by every run.

<!-- results:start -->
| Run | Validation | Leak guard | Bets | Realized ROI +/- SE | Expected ROI under the truth | Mean CLV | Bets with CLV > 0 | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|---|
| Honest walk-forward | chronological, data before the bet only | pass | 961 | +10.2% +/- 5.7% | -0.4% | -1.6% | 36% | 0.6268 | 1.0419 |
| Leak: closing price in features | chronological, closing odds as a feature | FAIL | 462 | +4.4% +/- 8.9% | +5.9% | +3.7% | 84% | 0.6146 | 1.0245 |
| Leak: random K-fold | random 5-fold instead of chronological | FAIL | 948 | +3.1% +/- 5.7% | -0.4% | -1.5% | 38% | 0.6246 | 1.0382 |

| Forecaster | Brier | Log loss |
|---|---|---|
| Model (honest walk-forward) | 0.6268 | 1.0419 |
| Opening market, multiplicative de-vig | 0.6147 | 1.0247 |
| Closing market, multiplicative de-vig | 0.6148 | 1.0249 |
| Closing market, shin de-vig | 0.6150 | 1.0252 |
| Closing market, power de-vig | 0.6151 | 1.0253 |
| Truth (synthetic only) | 0.6145 | 1.0244 |
<!-- results:end -->

Brier is the multiclass sum convention: squared error summed over home, draw and away, 0 perfect, 2 worst (the over/under scores in `RESULTS.md` are also shown halved, in the one-outcome 0-1 convention). Log loss is in nats.

How to read it:

- **The honest run "made" +10.2% on 961 bets, and it has no edge.** Scored against the true probabilities its bets are worth -0.4%, only 36% of them beat the closing line, and its mean CLV is -1.6%. The realized ROI is a draw 1.9 standard errors above its expectation. This is why a betting model is judged by CLV and calibration, not by its ROI over a season.
- **The closing-price leak creates an edge that cannot be bet.** Given the closing price as a feature, a stacker fitted on the sample puts weight 0.87 on it (fitted in-sample, on the very results it is then scored on: a second leak of its own); 84% of its bets then beat the close, mean CLV +3.7%, expected ROI +5.9%. All of it comes from knowing where the line will close, which nobody knows when the bet is placed. The guard fails it: the closing line is timestamped at kickoff. Its realized ROI (+4.4% ± 8.9% on 462 bets) is below the honest run's lucky draw: on a few hundred bets, result noise hides even a real six-point edge.
- **Random K-fold flatters the scores.** Ratings fitted on shuffled folds see matches played after the one they predict: Brier 0.6268 → 0.6246, log loss 1.0419 → 1.0382. On this league it does not move the expected ROI (-0.4% both); how much it does depends on how fast team strength drifts (2% a round here). The guard fails it: its features include results from after kickoff.
- **The market is the benchmark, and it is hard to beat.** The de-vigged closing line scores within 0.0003 Brier of the truth; the honest model is 0.012 behind. The three de-vig methods tie here because the synthetic books apply their margin proportionally; on real books, which load the margin on longshots, Shin and power separate from the proportional method.

`RESULTS.md` also carries the over/under 2.5 scores and the reliability table (forecast bins against observed frequencies) for the model and the closing market.

## The leak guard

Every prediction carries two timestamps: `decision_time`, when the price is taken, and `feature_time`, the latest timestamp of any data it used. Results are known at kickoff + 2 hours, the pre-closing price is taken a day before kickoff, the closing line exists only at kickoff. `check_no_leak` raises if any `feature_time >= decision_time` (equality included) or any `decision_time >= kickoff`. The tests run it on all three runs, and also scramble every result after a match's decision time to prove the honest prediction for that match does not move while the random K-fold one does.

## The odds math

| Need | Function | Checked against |
|---|---|---|
| Remove the margin, 2-way and 3-way | `devig.multiplicative`, `devig.shin`, `devig.power` | exact fractions; Shin's and power's own forward models inverted; Shin on two outcomes equals splitting the margin equally |
| Win-loss probability | `poisson.score_matrix` → `poisson.outcome_probs` | double sums of Poisson terms |
| Total points (match and team) | `poisson.over_under`, `poisson.team_total`, `poisson.expected_total` | a sum of independent Poissons is Poisson; pushes on whole-number lines |
| Odds of winning a bet | `poisson.fair_odds`, `poisson.expected_value` | 1 / p and p × odds − 1 |
| Calibration | `calibration.brier`, `log_loss`, `reliability_table`, `scorecard` | hand-worked rows |
| Closing line value | `clv.clv` (per bet), `clv.summarize` | taken odds × de-vigged closing probability − 1 |

## Data

- `data/synthetic_league.csv` — the committed fixture, generated by `python -m sbq.synthetic` from a fixed seed in football-data.co.uk's column layout, with the true probabilities next to the prices. Team strengths drift every round; goals are Poisson; the opening book is the truth plus noise plus a 3% margin; the closing book keeps 40% of the opening error, adds less fresh noise and a 2.5% margin. The tests and the table above run on it, offline.
- `data/fetch_football_data.py` — downloads the 2023/24 English Premier League season from [football-data.co.uk](https://www.football-data.co.uk/data.php) (Pinnacle pre-closing and closing prices) into the git-ignored `data/raw/`, and refuses it unless its SHA-256 matches the pinned one. football-data.co.uk states that its data "is intended for private individuals only, NOT commerical or data training products using automated bots/scrapers/AI", so this repository does not redistribute the file or publish results computed from it. The loader also reads the market average (`Avg*`) where Pinnacle's price is missing, and older seasons without a kickoff time.

## Layout

- `sbq/devig.py` — margin removal: multiplicative, Shin, power.
- `sbq/poisson.py` — scoreline matrix, win/draw/loss, match and team totals, fair odds, expected value.
- `sbq/calibration.py` — Brier, log loss, reliability table, scorecard.
- `sbq/clv.py` — closing line value per bet and aggregated.
- `sbq/data.py` — football-data.co.uk CSV loader, with a timestamp on every record.
- `sbq/synthetic.py` — the synthetic league.
- `sbq/backtest.py` — ratings, walk-forward predictions, the two leaks, the guard, bet settlement.
- `sbq/report.py`, `sbq/__main__.py` — `python -m sbq` renders `RESULTS.md`.

Dependencies: pandas, numpy, scipy. No betting or backtesting framework, on purpose: in an audit every line has to be readable.

## Limits, stated plainly

- The headline numbers are synthetic. They show the mechanisms (what each leak does, how noisy ROI is, what CLV measures) where the truth is known; they say nothing about the edge available in any real market.
- The rating model is a simple attack/defence average with time decay, not a fitted Dixon-Coles model; it is there to be audited, not to win. Independent Poisson suits goals; high-scoring sports (basketball points) need a normal or negative-binomial model of totals.
- The synthetic league plays each round at a single kickoff time, so on it a same-day leak (a 12:30 result entering the 15:00 prediction) could not show. A dedicated test builds two Saturday matches and checks the earlier result never reaches the later one, whose price was taken on Friday.
- The pre-closing price is assumed to be taken one day before kickoff. football-data.co.uk collects it on Friday or Tuesday afternoon before a round, so for some matches the real lead is longer.
- Flat one-unit stakes. Kelly or any other staking is out of scope.
- One CLV definition (taken odds against the de-vigged close). No in-play markets, no Asian handicap, no limits or bet acceptance.
