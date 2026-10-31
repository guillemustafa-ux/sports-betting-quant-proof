# Results

Data: `data/synthetic_league.csv`, 1140 matches, 1080 scored (both teams need 6 known results first; every run is scored on the same matches).
Bets: one unit at the pre-closing price on the outcome with the highest expected value, when it exceeds 3%. CLV against the closing line, multiplicative de-vig. Mean overround: opening 3.00%, closing 2.50%.

## Backtest audit

| Run | Validation | Leak guard | Bets | Realized ROI +/- SE | Expected ROI under the truth | Mean CLV | Bets with CLV > 0 | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|---|
| Honest walk-forward | chronological, data before the bet only | pass | 961 | +10.2% +/- 5.7% | -0.4% | -1.6% | 36% | 0.6268 | 1.0419 |
| Leak: closing price in features | chronological, closing odds as a feature | FAIL | 462 | +4.4% +/- 8.9% | +5.9% | +3.7% | 84% | 0.6146 | 1.0245 |
| Leak: random K-fold | random 5-fold instead of chronological | FAIL | 948 | +3.1% +/- 5.7% | -0.4% | -1.5% | 38% | 0.6246 | 1.0382 |

The closing-price stacker put weight 0.87 on the leaked column.

## Calibration against the market (1X2, honest model's matches)

| Forecaster | Brier | Log loss |
|---|---|---|
| Model (honest walk-forward) | 0.6268 | 1.0419 |
| Opening market, multiplicative de-vig | 0.6147 | 1.0247 |
| Closing market, multiplicative de-vig | 0.6148 | 1.0249 |
| Closing market, shin de-vig | 0.6150 | 1.0252 |
| Closing market, power de-vig | 0.6151 | 1.0253 |
| Truth (synthetic only) | 0.6145 | 1.0244 |

Over/under 2.5 goals on 1080 matches: model Brier 0.4520, closing market Brier 0.4362.

## Reliability: honest model vs. closing market (every outcome forecast, 10 bins)

| Forecast bin | Model: n | Model: mean forecast | Model: observed | Close: n | Close: mean forecast | Close: observed |
|---|---|---|---|---|---|---|
| 0.0-0.1 | 25 | 0.074 | 0.080 | 31 | 0.081 | 0.097 |
| 0.1-0.2 | 364 | 0.168 | 0.206 | 425 | 0.166 | 0.207 |
| 0.2-0.3 | 1311 | 0.248 | 0.253 | 1187 | 0.250 | 0.247 |
| 0.3-0.4 | 621 | 0.345 | 0.353 | 714 | 0.345 | 0.336 |
| 0.4-0.5 | 450 | 0.446 | 0.411 | 418 | 0.448 | 0.445 |
| 0.5-0.6 | 295 | 0.546 | 0.546 | 278 | 0.542 | 0.536 |
| 0.6-0.7 | 132 | 0.640 | 0.636 | 137 | 0.644 | 0.628 |
| 0.7-0.8 | 35 | 0.743 | 0.486 | 40 | 0.739 | 0.650 |
| 0.8-0.9 | 6 | 0.819 | 0.667 | 10 | 0.832 | 0.900 |
| 0.9-1.0 | 1 | 0.902 | 1.000 | 0 |  |  |
