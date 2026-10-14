"""
Tests for src/data/cache.py.

Requirements from phase1_plan.md §6:
- Second request for a cached range makes zero calls
- Newer end date fetches only tail + overlap
- Overlap mismatch triggers full refetch
- Corrupt Parquet / metadata mismatch triggers refetch
- no_data respected until retry_after
- Today's partial bar never stored (tested via yfinance_provider; here we test
  via FakeProvider which generates complete bars)
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.cache import CacheManager
from src.data.calendar import TradingCalendar
from src.data.provider import empty_frame
from tests.fakes.fake_provider import FakeProvider


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _weekdays(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def make_cache(tmp_path: Path, calendar: TradingCalendar) -> CacheManager:
    return CacheManager(
        tmp_path / "cache",
        calendar=calendar,
        overlap_sessions=5,
        no_data_retry_days=30,
        adj_tolerance=0.02,
        config_start_date=date(2020, 1, 1),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_cached_range_makes_no_second_call(tmp_path, calendar):
    """A second update_symbol call for a fresh symbol returns 'skipped'."""
    cache = make_cache(tmp_path, calendar)
    p = FakeProvider(["AAPL"])

    outcome1 = cache.update_symbol(p, "AAPL")
    assert outcome1 == "updated"
    assert p.call_count["AAPL"] == 1

    # Second call: should be skipped (already fresh)
    outcome2 = cache.update_symbol(p, "AAPL")
    assert outcome2 == "skipped"
    assert p.call_count["AAPL"] == 1  # no extra call


def test_incremental_fetch_only_tail(tmp_path, calendar):
    """
    After an initial fetch, moving the end date forward fetches only
    the tail (overlap + new rows), not the full history.
    """
    # Initial calendar covers up to 2022-12-31
    cal_short = TradingCalendar.from_dates(_weekdays(date(2020, 1, 1), date(2022, 12, 31)))
    cache = CacheManager(
        tmp_path / "cache",
        calendar=cal_short,
        overlap_sessions=5,
        no_data_retry_days=30,
        adj_tolerance=0.02,
        config_start_date=date(2020, 1, 1),
    )
    p = FakeProvider(["MSFT"])
    cache.update_symbol(p, "MSFT")
    calls_after_first = p.call_count["MSFT"]

    # Extend calendar to 2023-06-30 → cache is stale
    cal_long = TradingCalendar.from_dates(_weekdays(date(2020, 1, 1), date(2023, 6, 30)))
    cache.set_calendar(cal_long)

    cache.update_symbol(p, "MSFT")
    # Provider was called exactly once more (tail fetch)
    assert p.call_count["MSFT"] == calls_after_first + 1


def test_overlap_mismatch_triggers_full_refetch(tmp_path):
    """If adj_close in the overlap window changes beyond tolerance, full refetch."""
    # Phase 1: populate cache with short calendar
    cal_short = TradingCalendar.from_dates(_weekdays(date(2020, 1, 1), date(2022, 12, 30)))
    cache = CacheManager(
        tmp_path / "cache",
        calendar=cal_short,
        overlap_sessions=5,
        no_data_retry_days=30,
        adj_tolerance=0.02,
        config_start_date=date(2020, 1, 1),
    )
    p1 = FakeProvider(["XYZ"], base_price=100.0)
    cache.update_symbol(p1, "XYZ")

    # Phase 2: extend calendar so cache is stale, then use a provider whose
    # adj_close is very different (simulates a retroactive dividend adjustment)
    cal_long = TradingCalendar.from_dates(_weekdays(date(2020, 1, 1), date(2023, 6, 30)))
    cache.set_calendar(cal_long)

    p2 = FakeProvider(["XYZ"], base_price=50.0)  # 50% change >> 2% tolerance
    outcome = cache.update_symbol(p2, "XYZ")

    assert outcome == "refetched"
    assert p2.call_count.get("XYZ", 0) >= 1


def test_corrupt_parquet_triggers_refetch(tmp_path, calendar):
    """A corrupt Parquet file causes load() to return None and re-fetch."""
    cache = make_cache(tmp_path, calendar)
    p = FakeProvider(["ABC"])
    cache.update_symbol(p, "ABC")

    # Corrupt the Parquet file
    parquet_path = cache._parquet_path("fake", "ABC")
    parquet_path.write_bytes(b"not a parquet file")

    # load() should return None
    assert cache.load("fake", "ABC") is None

    # update_symbol should refetch
    outcome = cache.update_symbol(p, "ABC")
    assert outcome in ("updated", "refetched")
    assert cache.load("fake", "ABC") is not None


def test_metadata_last_date_mismatch_invalidates(tmp_path, calendar):
    """If the Parquet last date doesn't match metadata, load() returns None."""
    cache = make_cache(tmp_path, calendar)
    p = FakeProvider(["DEF"])
    cache.update_symbol(p, "DEF")

    # Tamper with metadata: set last_date to a wrong value
    cache._upsert_meta("fake", "DEF", last_date="2099-01-01")
    assert cache.load("fake", "DEF") is None


def test_no_data_respected_until_retry_after(tmp_path, calendar):
    """A no_data symbol is not re-requested until retry_after."""
    cache = make_cache(tmp_path, calendar)
    p = FakeProvider()  # "ZZZ" not in symbols → returns empty
    p.add_symbol("")    # don't add ZZZ

    outcome1 = cache.update_symbol(p, "ZZZ")
    assert outcome1 == "no_data"
    calls1 = p.call_count.get("ZZZ", 0)

    outcome2 = cache.update_symbol(p, "ZZZ")
    assert outcome2 == "skipped"
    # Provider should NOT have been called again
    assert p.call_count.get("ZZZ", 0) == calls1


def test_load_returns_none_for_unknown_symbol(tmp_path, calendar):
    cache = make_cache(tmp_path, calendar)
    assert cache.load("fake", "UNKNOWN") is None


def test_update_and_load_roundtrip(tmp_path, calendar):
    """update_symbol followed by load returns a non-empty DataFrame."""
    cache = make_cache(tmp_path, calendar)
    p = FakeProvider(["SPY"])
    cache.update_symbol(p, "SPY")

    df = cache.load("fake", "SPY")
    assert df is not None
    assert not df.empty
    assert "close" in df.columns
    assert "adj_close" in df.columns


def test_run_logging(tmp_path, calendar):
    """start_run and finish_run persist to SQLite without error."""
    cache = make_cache(tmp_path, calendar)
    run_id = cache.start_run("fake")
    assert isinstance(run_id, str)
    cache.finish_run(run_id, symbols_updated=5, symbols_no_data=1)
