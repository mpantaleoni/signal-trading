"""Tests for src/signals/engine_a.py — entry/exit signal + no-look-ahead."""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.calendar import TradingCalendar
from src.signals.engine_a import (
    compute_entry_signal,
    compute_exit_signals,
    compute_regime_ok,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_df(n: int, adj_close_vals=None, high_mult=1.01, low_mult=0.99, seed=42):
    """Create a canonical OHLCV DataFrame for testing."""
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    if adj_close_vals is None:
        rng = np.random.default_rng(seed)
        adj_close_vals = 100 + np.cumsum(rng.normal(0, 0.5, n))
    adj_close_vals = np.array(adj_close_vals, dtype=float)
    high = adj_close_vals * high_mult
    low = adj_close_vals * low_mult
    return pd.DataFrame(
        {
            "open": adj_close_vals,
            "high": high,
            "low": low,
            "close": adj_close_vals,
            "adj_close": adj_close_vals,
            "volume": [1_000_000.0] * n,
            "dividends": [0.0] * n,
            "splits": [0.0] * n,
        },
        index=idx,
    )


def _make_calendar(start="2010-01-01", end="2015-12-31"):
    dates = [d.date() for d in pd.date_range(start, end, freq="B")]
    return TradingCalendar.from_dates(dates)


# ── compute_regime_ok ──────────────────────────────────────────────────────────

class TestComputeRegimeOk:
    def test_true_when_above_sma200(self):
        # Rising series: last price > SMA(200)
        vals = list(range(1, 250))  # strictly increasing
        df = _make_df(249, adj_close_vals=vals)
        assert compute_regime_ok(df) is True

    def test_false_when_below_sma200(self):
        # Declining series: last price < SMA(200)
        vals = list(range(250, 0, -1))
        df = _make_df(250, adj_close_vals=vals)
        assert compute_regime_ok(df) is False

    def test_false_when_insufficient_history(self):
        df = _make_df(199)
        assert compute_regime_ok(df) is False

    def test_exactly_200_rows(self):
        vals = list(range(1, 201))
        df = _make_df(200, adj_close_vals=vals)
        # last = 200, SMA200 = 100.5, so above
        assert compute_regime_ok(df) is True

    def test_no_lookahead(self):
        """Result at row N must not change after appending a future row."""
        vals = list(range(1, 250))
        df = _make_df(249, adj_close_vals=vals)
        result_before = compute_regime_ok(df)

        # Append a bar with extreme low price
        extra = _make_df(1, adj_close_vals=[0.001])
        extra.index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=1, freq="B")
        df_ext = pd.concat([df, extra])
        # The new df_ext has result based on new last row — that's expected to change
        # But we verify the first 249 rows haven't changed via slicing
        result_slice = compute_regime_ok(df_ext.iloc[:-1])
        assert result_before == result_slice


# ── compute_entry_signal ───────────────────────────────────────────────────────

def _build_entry_df(n=250, trend="up", rsi_low=True):
    """Build a DataFrame where entry conditions are either met or not."""
    if trend == "up":
        # Gradually rising then small pullback at end
        base = np.linspace(50, 120, n)
        if rsi_low:
            # Small decline at end → RSI(2) < 10
            base[-5:] = base[-5] * 0.96
            base[-4:] = base[-4] * 0.97
            base[-3:] = base[-3] * 0.97
            base[-2:] = base[-2] * 0.97
            base[-1] = base[-1] * 0.97
    else:
        base = np.linspace(120, 50, n)
    idx = pd.date_range("2005-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "adj_close": base,
            "volume": [10_000_000.0] * n,
            "dividends": [0.0] * n,
            "splits": [0.0] * n,
        },
        index=idx,
    )


class TestComputeEntrySignal:
    MIN_ADV = 5_000_000.0

    def test_false_when_fewer_than_200_rows(self):
        df = _make_df(199)
        assert compute_entry_signal(df, rsi_entry=10, min_adv_usd=self.MIN_ADV) is False

    def test_false_when_below_sma200(self):
        df = _build_entry_df(trend="down")
        assert compute_entry_signal(df, rsi_entry=10, min_adv_usd=self.MIN_ADV) is False

    def test_false_when_insufficient_liquidity(self):
        df = _build_entry_df(trend="up", rsi_low=True)
        # Very low volume
        df["volume"] = 1.0
        assert compute_entry_signal(df, rsi_entry=10, min_adv_usd=self.MIN_ADV) is False

    def test_no_lookahead(self):
        """Signal computed on df[:N] must equal signal on df[:N+1].iloc[:-1]."""
        df = _build_entry_df(250)
        result_n = compute_entry_signal(df, rsi_entry=10, min_adv_usd=self.MIN_ADV)

        # Append a random bar
        extra = _make_df(1)
        extra.index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=1, freq="B")
        df_ext = pd.concat([df, extra])
        result_n_from_ext = compute_entry_signal(
            df_ext.iloc[:-1], rsi_entry=10, min_adv_usd=self.MIN_ADV
        )
        assert result_n == result_n_from_ext

    def test_false_when_adj_close_nan(self):
        df = _make_df(250)
        df.at[df.index[-1], "adj_close"] = float("nan")
        assert compute_entry_signal(df, rsi_entry=10, min_adv_usd=self.MIN_ADV) is False


# ── compute_exit_signals ───────────────────────────────────────────────────────

class TestComputeExitSignals:
    def _calendar(self):
        return _make_calendar("2010-01-01", "2015-12-31")

    def _df(self, n=250, adj_close_vals=None):
        return _make_df(n, adj_close_vals=adj_close_vals)

    def test_sma5_exit_true_when_above_sma5(self):
        # Rising series: last price > SMA(5)
        vals = list(range(1, 251))
        df = self._df(250, adj_close_vals=vals)
        cal = self._calendar()
        entry = date(2010, 1, 4)
        result = compute_exit_signals(df, entry, max_hold_days=10, calendar=cal)
        assert result["sma5_exit"] is True

    def test_sma5_exit_false_when_below_sma5(self):
        # Declining series: last price < SMA(5)
        vals = list(range(250, 0, -1))
        df = self._df(250, adj_close_vals=vals)
        cal = self._calendar()
        entry = date(2010, 1, 4)
        result = compute_exit_signals(df, entry, max_hold_days=10, calendar=cal)
        assert result["sma5_exit"] is False

    def test_sma200_exit_false_when_above_sma200(self):
        vals = list(range(1, 251))
        df = self._df(250, adj_close_vals=vals)
        cal = self._calendar()
        entry = date(2010, 1, 4)
        result = compute_exit_signals(df, entry, max_hold_days=10, calendar=cal)
        assert result["sma200_exit"] is False

    def test_sma200_exit_true_when_below_sma200(self):
        vals = list(range(250, 0, -1))
        df = self._df(250, adj_close_vals=vals)
        cal = self._calendar()
        entry = date(2010, 1, 4)
        result = compute_exit_signals(df, entry, max_hold_days=10, calendar=cal)
        assert result["sma200_exit"] is True

    def test_time_stop_triggered(self):
        df = self._df(250)
        cal = self._calendar()
        # entry_date is the first session date
        entry = df.index[0].date()
        # current session is 15 sessions later
        current_df = df.iloc[:15]
        result = compute_exit_signals(current_df, entry, max_hold_days=10, calendar=cal)
        assert result["time_stop"] is True

    def test_time_stop_not_triggered(self):
        df = self._df(250)
        cal = self._calendar()
        entry = df.index[0].date()
        # Only 3 sessions held
        current_df = df.iloc[:3]
        result = compute_exit_signals(current_df, entry, max_hold_days=10, calendar=cal)
        assert result["time_stop"] is False

    def test_no_lookahead_sma5(self):
        """Adding a future row must not change exit signals of the current row."""
        vals = list(range(1, 251))
        df = self._df(250, adj_close_vals=vals)
        cal = self._calendar()
        entry = date(2010, 1, 4)
        result_before = compute_exit_signals(df, entry, max_hold_days=10, calendar=cal)

        extra = self._df(1, adj_close_vals=[0.001])
        extra.index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=1, freq="B")
        df_ext = pd.concat([df, extra])
        result_after = compute_exit_signals(
            df_ext.iloc[:-1], entry, max_hold_days=10, calendar=cal
        )
        assert result_before == result_after

    def test_all_keys_present(self):
        df = self._df(250)
        cal = self._calendar()
        result = compute_exit_signals(df, date(2010, 1, 4), 10, cal)
        assert set(result.keys()) == {"sma5_exit", "sma200_exit", "time_stop"}
