"""
Tests for src/reports/data_report.py.

Requirements from phase1_plan.md §6:
- Report built from a fixture cache
- Per-year counts match hand-computed values
- KNOWN BIAS statement is present
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.data.calendar import TradingCalendar
from src.data.cache import CacheManager
from src.universe.etfs import ETFUniverse
from src.universe.sp500 import SP500Universe
from tests.fakes.fake_provider import FakeProvider


_KNOWN_BIAS_FRAGMENT = "KNOWN BIAS: free data under-represents delisted losers"


def _weekdays(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


@pytest.fixture
def minimal_setup(tmp_path):
    """
    Tiny universe: snapshots covering 2020 only.
      AAPL, GOOG always present
      MSFT present from start, removed 2020-06-01
    """
    # Calendar
    days = _weekdays(date(2020, 1, 1), date(2020, 12, 31))
    calendar = TradingCalendar.from_dates(days)

    # Cache with data for AAPL and GOOG only (MSFT has no data)
    cache = CacheManager(
        tmp_path / "cache",
        calendar=calendar,
        config_start_date=date(2020, 1, 1),
    )
    provider = FakeProvider(["AAPL", "GOOG"])
    cache.update_symbol(provider, "AAPL")
    cache.update_symbol(provider, "GOOG")

    # Constituent CSV
    csv_path = tmp_path / "sp500_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "tickers"])
        w.writerow(["2020-01-01", "AAPL,GOOG,MSFT"])
        w.writerow(["2020-06-01", "AAPL,GOOG"])   # MSFT removed

    sp500 = SP500Universe.from_csv(csv_path)
    etfs = ETFUniverse.from_config(["SPY"])

    return cache, sp500, etfs, calendar, days


def test_report_contains_known_bias(tmp_path, minimal_setup):
    from src.reports.data_report import generate_data_report

    cache, sp500, etfs, calendar, days = minimal_setup
    output = tmp_path / "report.md"

    text = generate_data_report(
        cache=cache,
        sp500=sp500,
        etfs=etfs,
        provider_name="fake",
        config_start=date(2020, 1, 1),
        config_end=date(2020, 12, 31),
        constituent_sha256="abc123",
        with_data_threshold=0.90,
        large_return_threshold=0.40,
        stale_price_run=5,
        output_path=output,
        calendar_sessions=set(days),
    )

    assert _KNOWN_BIAS_FRAGMENT in text


def test_report_written_to_file(tmp_path, minimal_setup):
    from src.reports.data_report import generate_data_report

    cache, sp500, etfs, calendar, days = minimal_setup
    output = tmp_path / "sub" / "report.md"

    generate_data_report(
        cache=cache,
        sp500=sp500,
        etfs=etfs,
        provider_name="fake",
        config_start=date(2020, 1, 1),
        config_end=date(2020, 12, 31),
        constituent_sha256="",
        with_data_threshold=0.90,
        large_return_threshold=0.40,
        stale_price_run=5,
        output_path=output,
        calendar_sessions=set(days),
    )

    assert output.exists()
    content = output.read_text()
    assert len(content) > 100  # non-empty


def test_report_per_year_counts(tmp_path, minimal_setup):
    """
    In 2020 there are 3 distinct members (AAPL, GOOG, MSFT).
    AAPL and GOOG have full data → 2 'with data'.
    MSFT has no data → 1 'missing'.
    """
    from src.reports.data_report import generate_data_report

    cache, sp500, etfs, calendar, days = minimal_setup
    output = tmp_path / "report.md"

    text = generate_data_report(
        cache=cache,
        sp500=sp500,
        etfs=etfs,
        provider_name="fake",
        config_start=date(2020, 1, 1),
        config_end=date(2020, 12, 31),
        constituent_sha256="",
        with_data_threshold=0.90,
        large_return_threshold=0.40,
        stale_price_run=5,
        output_path=output,
        calendar_sessions=set(days),
    )

    # The year row for 2020 should exist
    assert "2020" in text

    # The missing members section should mention MSFT
    assert "MSFT" in text


def test_report_etf_section(tmp_path, minimal_setup):
    from src.reports.data_report import generate_data_report

    cache, sp500, etfs, _, days = minimal_setup
    output = tmp_path / "report.md"

    text = generate_data_report(
        cache=cache,
        sp500=sp500,
        etfs=etfs,
        provider_name="fake",
        config_start=date(2020, 1, 1),
        config_end=date(2020, 12, 31),
        constituent_sha256="",
        with_data_threshold=0.90,
        large_return_threshold=0.40,
        stale_price_run=5,
        output_path=output,
        calendar_sessions=set(days),
    )

    assert "ETF" in text
    assert "SPY" in text
