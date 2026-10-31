"""python -m sbq [--csv PATH] [--out PATH]: run the three backtests and write the results."""

from __future__ import annotations

import argparse
import pathlib
import sys

from sbq import backtest, data, report

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m sbq", description=__doc__)
    parser.add_argument("--csv", default="data/synthetic_league.csv", help="football-data.co.uk style CSV")
    parser.add_argument("--out", default="RESULTS.md", help="where to write the Markdown results")
    args = parser.parse_args(argv)
    matches = data.load_matches(args.csv)
    runs = {mode: backtest.run(matches, mode) for mode in backtest.MODES}
    text = report.render(matches, runs, source=pathlib.Path(args.csv).as_posix())
    pathlib.Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
