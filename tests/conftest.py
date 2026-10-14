"""Shared fixtures for all tests."""
from __future__ import annotations

import csv
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.calendar import TradingCalendar
from src.data.cache import CacheManager
from tests.fakes.fake_provider import FakeProvider


def _weekdays(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


@pytest.fixture
def calendar():
    """A calendar covering 2020-01-01 to 2024-12-31 (Mon–Fri)."""
    days = _weekdays(date(2020, 1, 1), date(2024, 12, 31))
    return TradingCalendar.from_dates(days)


@pytest.fixture
def tmp_cache(tmp_path, calendar):
    """A fresh CacheManager backed by a temp directory."""
    return CacheManager(
        tmp_path / "cache",
        calendar=calendar,
        overlap_sessions=5,
        no_data_retry_days=30,
        adj_tolerance=0.02,
        config_start_date=date(2020, 1, 1),
    )


@pytest.fixture
def fake_provider():
    """A FakeProvider with two symbols."""
    p = FakeProvider(["AAPL", "GOOG"])
    return p


@pytest.fixture
def sp500_csv(tmp_path) -> Path:
    """
    A minimal synthetic S&P 500 constituent CSV for offline tests.

    Snapshots:
      2020-01-01: AAPL, GOOG, MSFT
      2020-06-01: AAPL, GOOG, MSFT, AMZN   (AMZN added)
      2021-01-01: AAPL, GOOG, AMZN          (MSFT removed)
    """
    path = tmp_path / "sp500_history.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "tickers"])
        writer.writerow(["2020-01-01", "AAPL,GOOG,MSFT"])
        writer.writerow(["2020-06-01", "AAPL,GOOG,MSFT,AMZN"])
        writer.writerow(["2021-01-01", "AAPL,GOOG,AMZN"])
    return path
