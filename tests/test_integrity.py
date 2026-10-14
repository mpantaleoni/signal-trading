"""
Tests for src/data/integrity.py — passing and failing synthetic series for each check.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.integrity import (
    IntegrityFailure,
    check_adjustment_jumps,
    check_missing_bars,
    check_ohlc_consistency,
    check_price_validity,
    check_stale_prices,
    check_zero_volume,
    run_all_checks,
)


# ──────────────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_df(dates: list[date], **overrides) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame for the given dates.

    Default prices increment by 1.0 per row so consecutive closes are
    never identical (avoids spurious stale_prices failures in clean-data tests).
    """
    rows = []
    for i, d in enumerate(dates):
        price = 100.0 + i  # incrementing avoids stale_prices trigger
        row = {
            "open": price,
            "high": price + 2.0,
            "low": price - 2.0,
            "close": price,
            "adj_close": price,
            "volume": 1_000_000.0,
            "dividends": 0.0,
            "splits": 0.0,
        }
        row.update({k: v[i] if isinstance(v, list) else v
                    for k, v in overrides.items()})
        rows.append(row)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    return df


def _dates(start: date = date(2022, 1, 3), n: int = 10) -> list[date]:
    """Generate n weekdays starting from start."""
    result = []
    d = start
    while len(result) < n:
        if d.weekday() < 5:
            result.append(d)
        d += timedelta(days=1)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# check_missing_bars
# ──────────────────────────────────────────────────────────────────────────────

class TestMissingBars:
    def test_no_failures_when_all_present(self):
        days = _dates(n=5)
        df = _make_df(days)
        failures = check_missing_bars("X", df, set(days))
        assert failures == []

    def test_detects_missing_session(self):
        days = _dates(n=5)
        calendar = set(days)
        # Remove the middle bar
        df = _make_df([days[0], days[1], days[3], days[4]])
        failures = check_missing_bars("X", df, calendar)
        assert len(failures) == 1
        assert failures[0].check == "missing_bar"
        assert failures[0].date == days[2]

    def test_empty_df_no_failures(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "adj_close",
                                   "volume", "dividends", "splits"])
        failures = check_missing_bars("X", df, set(_dates()))
        assert failures == []


# ──────────────────────────────────────────────────────────────────────────────
# check_price_validity
# ──────────────────────────────────────────────────────────────────────────────

class TestPriceValidity:
    def test_no_failures_for_clean_data(self):
        df = _make_df(_dates())
        assert check_price_validity("X", df) == []

    def test_detects_zero_close(self):
        days = _dates(n=3)
        df = _make_df(days, close=[100.0, 0.0, 100.0])
        failures = check_price_validity("X", df)
        assert any(f.check == "price_validity" and "close" in f.detail for f in failures)

    def test_detects_negative_open(self):
        days = _dates(n=2)
        df = _make_df(days, open=[100.0, -5.0])
        failures = check_price_validity("X", df)
        assert any("open" in f.detail for f in failures)

    def test_detects_nan_adj_close(self):
        import numpy as np
        days = _dates(n=2)
        df = _make_df(days, adj_close=[100.0, float("nan")])
        failures = check_price_validity("X", df)
        assert any("adj_close" in f.detail for f in failures)


# ──────────────────────────────────────────────────────────────────────────────
# check_adjustment_jumps
# ──────────────────────────────────────────────────────────────────────────────

class TestAdjustmentJumps:
    def test_no_failures_for_normal_data(self):
        days = _dates(n=5)
        df = _make_df(days)
        assert check_adjustment_jumps("X", df) == []

    def test_detects_split_like_jump_no_split_recorded(self):
        days = _dates(n=3)
        # close doubles with no split
        df = _make_df(days, close=[100.0, 200.0, 200.0], splits=[0.0, 0.0, 0.0])
        failures = check_adjustment_jumps("X", df)
        assert any(f.check == "split_like_jump" for f in failures)

    def test_no_flag_if_split_recorded(self):
        days = _dates(n=3)
        # close doubles AND split is recorded
        df = _make_df(days, close=[100.0, 200.0, 200.0], splits=[0.0, 2.0, 0.0])
        failures = check_adjustment_jumps("X", df)
        assert not any(f.check == "split_like_jump" for f in failures)

    def test_detects_large_adj_return(self):
        days = _dates(n=3)
        # adj_close jumps 50% in one day
        df = _make_df(days, adj_close=[100.0, 150.0, 150.0])
        failures = check_adjustment_jumps("X", df, large_return_threshold=0.40)
        assert any(f.check == "large_adj_return" for f in failures)

    def test_no_flag_for_small_return(self):
        days = _dates(n=3)
        df = _make_df(days, adj_close=[100.0, 101.0, 102.0])
        assert check_adjustment_jumps("X", df, large_return_threshold=0.40) == []


# ──────────────────────────────────────────────────────────────────────────────
# check_stale_prices
# ──────────────────────────────────────────────────────────────────────────────

class TestStalePrices:
    def test_no_failures_for_varying_prices(self):
        days = _dates(n=10)
        closes = [100.0 + i for i in range(10)]
        df = _make_df(days, close=closes)
        assert check_stale_prices("X", df, run=5) == []

    def test_detects_run_of_5(self):
        days = _dates(n=7)
        closes = [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        df = _make_df(days, close=closes)
        failures = check_stale_prices("X", df, run=5)
        assert len(failures) >= 1
        assert all(f.check == "stale_prices" for f in failures)

    def test_no_flag_for_run_of_4_when_threshold_is_5(self):
        days = _dates(n=6)
        closes = [100.0, 100.0, 100.0, 100.0, 99.0, 100.0]
        df = _make_df(days, close=closes)
        assert check_stale_prices("X", df, run=5) == []


# ──────────────────────────────────────────────────────────────────────────────
# check_ohlc_consistency (D5)
# ──────────────────────────────────────────────────────────────────────────────

class TestOHLCConsistency:
    def test_no_failures_for_valid_bar(self):
        df = _make_df(_dates(n=3))
        assert check_ohlc_consistency("X", df) == []

    def test_detects_high_less_than_low(self):
        days = _dates(n=2)
        # high=98, low=102 → high < low
        df = _make_df(days, high=[102.0, 98.0], low=[98.0, 102.0])
        failures = check_ohlc_consistency("X", df)
        assert any("high < low" in f.detail for f in failures)

    def test_detects_high_less_than_close(self):
        days = _dates(n=2)
        df = _make_df(days, high=[102.0, 99.0], close=[100.0, 103.0])
        failures = check_ohlc_consistency("X", df)
        assert any("high < close" in f.detail for f in failures)

    def test_detects_low_greater_than_open(self):
        days = _dates(n=2)
        df = _make_df(days, low=[98.0, 105.0], open=[100.0, 100.0])
        failures = check_ohlc_consistency("X", df)
        assert any("low > open" in f.detail for f in failures)


# ──────────────────────────────────────────────────────────────────────────────
# check_zero_volume (D5)
# ──────────────────────────────────────────────────────────────────────────────

class TestZeroVolume:
    def test_no_failures_for_positive_volume(self):
        df = _make_df(_dates(n=3))
        assert check_zero_volume("X", df) == []

    def test_detects_zero_volume(self):
        days = _dates(n=3)
        df = _make_df(days, volume=[1_000_000, 0.0, 1_000_000])
        failures = check_zero_volume("X", df)
        assert len(failures) == 1
        assert failures[0].check == "zero_volume"
        assert failures[0].date == days[1]


# ──────────────────────────────────────────────────────────────────────────────
# run_all_checks
# ──────────────────────────────────────────────────────────────────────────────

class TestRunAllChecks:
    def test_clean_data_has_no_failures(self):
        days = _dates(n=10)
        df = _make_df(days)
        failures = run_all_checks("X", df, set(days))
        assert failures == []

    def test_combines_multiple_failures(self):
        days = _dates(n=5)
        calendar = set(days)
        # Introduce a missing bar and a zero volume
        df = _make_df([days[0], days[2], days[3], days[4]],
                      volume=[1_000_000, 1_000_000, 0.0, 1_000_000])
        failures = run_all_checks("X", df, calendar)
        checks = {f.check for f in failures}
        assert "missing_bar" in checks
        assert "zero_volume" in checks
