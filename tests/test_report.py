"""The rendered results: regenerate byte-identical, and the README quotes them exactly."""

import pandas as pd

from conftest import FIXTURE, ROOT
from sbq import __main__ as cli
from sbq import backtest as bt
from sbq import report


def _rendered(matches, runs):
    return report.render(matches, runs, source="data/synthetic_league.csv")


def test_committed_results_regenerate_byte_identical(matches, runs):
    assert (ROOT / "RESULTS.md").read_text(encoding="utf-8") == _rendered(matches, runs)


def test_readme_results_table_is_copied_verbatim_from_the_output(matches, runs):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- results:start -->")[1].split("<!-- results:end -->")[0]
    lines = [line for line in block.splitlines() if line.startswith("|")]
    assert len(lines) >= 5
    rendered = _rendered(matches, runs)
    for line in lines:
        assert line in rendered


def test_the_audit_table_flags_the_leaky_runs(matches, runs):
    text = _rendered(matches, runs)
    honest = next(line for line in text.splitlines() if line.startswith("| Honest walk-forward"))
    assert "| pass |" in honest
    for label in ("| Leak: closing price in features", "| Leak: random K-fold"):
        assert "| FAIL |" in next(line for line in text.splitlines() if line.startswith(label))


def test_the_report_states_the_brier_convention(matches, runs):
    text = _rendered(matches, runs)
    assert "Brier is the multiclass sum convention" in text
    ou = next(line for line in text.splitlines() if line.startswith("Over/under 2.5"))
    model, close = (float(x.split(",")[0].split(" ")[0]) for x in ou.split("Brier ")[1:3])
    assert f"{model / 2:.4f} and {close / 2:.4f} in the one-outcome 0-1 convention" in ou


def test_the_report_prints_the_config_the_runs_used(matches):
    cfg = bt.Config(min_team_matches=8, edge=0.05, method="shin")
    runs = {mode: bt.run(matches.iloc[:300], mode, cfg) for mode in bt.MODES}
    text = report.render(matches.iloc[:300], runs, source="x.csv")
    assert "both teams need 8 known results first" in text
    assert "when it exceeds 5%" in text
    assert "CLV against the closing line, shin de-vig" in text


def test_cli_writes_the_report_and_handles_data_without_truth(tmp_path, capsys):
    small = tmp_path / "small.csv"
    pd.read_csv(FIXTURE).iloc[:300].drop(columns=["TrueH", "TrueD", "TrueA", "TrueO25"]).to_csv(small, index=False)
    out = tmp_path / "out.md"
    cli.main(["--csv", str(small), "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert text == capsys.readouterr().out
    assert "Truth (synthetic only)" not in text
    honest = next(line for line in text.splitlines() if line.startswith("| Honest walk-forward"))
    assert "| n/a |" in honest  # no true probabilities, no expected ROI
