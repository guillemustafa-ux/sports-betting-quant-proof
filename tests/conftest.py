import pathlib

import pytest

from sbq import backtest as bt
from sbq import data

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "synthetic_league.csv"


@pytest.fixture(scope="session")
def matches():
    return data.load_matches(FIXTURE)


@pytest.fixture(scope="session")
def runs(matches):
    return {mode: bt.run(matches, mode) for mode in bt.MODES}
