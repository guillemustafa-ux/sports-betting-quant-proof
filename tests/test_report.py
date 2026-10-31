"""The rendered results: regenerate byte-identical, and the README quotes them exactly."""

import pandas as pd

from conftest import FIXTURE, ROOT
from sbq import __main__ as cli
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
