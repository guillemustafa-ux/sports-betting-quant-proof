"""Loader for football-data.co.uk style CSVs, the committed synthetic fixture, and the fetch checksum."""

import hashlib
import importlib.util
import pathlib

import pandas as pd
import pytest

from sbq import data, synthetic

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "synthetic_league.csv"

# Invented numbers in football-data.co.uk's column layout (with the BOM their files carry).
SAMPLE = """﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA,AvgH,AvgD,AvgA,P>2.5,P<2.5,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA,PC>2.5,PC<2.5
E0,12/08/2023,15:00,Alpha,Beta,2,1,H,2.10,3.40,3.60,2.05,3.30,3.50,1.95,1.90,2.00,3.50,3.80,1.98,3.40,3.70,1.90,1.95
E0,12/08/2023,12:30,Gamma,Delta,0,0,D,,,,1.80,3.60,4.50,2.10,1.75,1.85,3.60,4.40,1.82,3.55,4.30,2.05,1.80
E0,13/08/2023,16:30,Beta,Gamma,1,3,A,2.50,3.20,2.90,2.45,3.15,2.85,1.80,2.05,,,,,,,1.85,2.00
"""


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "E0.csv"
    path.write_text(SAMPLE, encoding="utf-8")
    return data.load_matches(path)


def test_loader_maps_columns_sorts_by_kickoff_and_reads_the_result(sample):
    assert list(sample.home) == ["Gamma", "Alpha"]  # 12:30 before 15:00; the third row has no closing 1X2
    first = sample.iloc[0]
    assert first.kickoff == pd.Timestamp("2023-08-12 12:30")
    assert (first.hg, first.ag, first.result) == (0, 0, 1)
    assert sample.iloc[1].result == 0
    assert list(sample.iloc[1][["close_h", "close_d", "close_a"]]) == [2.00, 3.50, 3.80]
    assert list(sample.iloc[1][["close_over", "close_under"]]) == [1.90, 1.95]


def test_loader_falls_back_to_market_average_when_pinnacle_is_missing(sample):
    assert list(sample.iloc[0][["open_h", "open_d", "open_a"]]) == [1.80, 3.60, 4.50]


def test_loader_times_every_record(sample):
    row = sample.iloc[1]
    assert row.close_time == row.kickoff
    assert row.open_time == row.kickoff - pd.Timedelta(days=1)
    assert row.result_time == row.kickoff + pd.Timedelta(hours=2)


def test_loader_defaults_kickoff_time_when_the_column_is_absent(tmp_path):
    lines = SAMPLE.lstrip("﻿").splitlines()
    header = lines[0].split(",")
    t = header.index("Time")
    rows = [",".join(c for i, c in enumerate(line.split(",")) if i != t) for line in lines]
    path = tmp_path / "old.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert data.load_matches(path).iloc[0].kickoff == pd.Timestamp("2023-08-12 15:00")


def test_committed_fixture_regenerates_byte_identical(tmp_path):
    out = tmp_path / "league.csv"
    synthetic.write_league(out)
    assert out.read_bytes() == FIXTURE.read_bytes()


def test_committed_fixture_loads_complete(tmp_path):
    m = data.load_matches(FIXTURE)
    assert len(m) == synthetic.SEASONS * synthetic.N_TEAMS * (synthetic.N_TEAMS - 1)
    assert m.kickoff.is_monotonic_increasing
    assert m[["open_h", "close_h", "open_over", "close_over"]].notna().all().all()
    assert (m[["open_h", "open_d", "open_a", "close_h", "close_d", "close_a"]] > 1).all().all()
    # the synthetic books carry the margins they were built with, within rounding
    over_open = (1 / m[["open_h", "open_d", "open_a"]]).sum(axis=1) - 1
    over_close = (1 / m[["close_h", "close_d", "close_a"]]).sum(axis=1) - 1
    assert over_open.mean() == pytest.approx(synthetic.MARGIN_OPEN, abs=0.003)
    assert over_close.mean() == pytest.approx(synthetic.MARGIN_CLOSE, abs=0.003)


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_football_data", ROOT / "data" / "fetch_football_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_script_accepts_only_the_pinned_checksum():
    fetch = _fetch_module()
    payload = b"Div,Date\n"
    fetch.verify(payload, hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="checksum"):
        fetch.verify(payload + b"x", hashlib.sha256(payload).hexdigest())
    assert len(fetch.SHA256) == 64
