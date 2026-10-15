"""Tests for src/signals/indicators.py — correctness + no-look-ahead."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.signals.indicators import atr, rsi, sma


# ── helpers ────────────────────────────────────────────────────────────────────

def _series(values, name="close"):
    idx = pd.date_range("2010-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, name=name, dtype=float)


def _ohlc(n=50, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    return (
        pd.Series(high, index=idx),
        pd.Series(low, index=idx),
        pd.Series(close, index=idx),
    )


# ── SMA ────────────────────────────────────────────────────────────────────────

class TestSma:
    def test_nan_for_insufficient_history(self):
        s = _series([1, 2, 3, 4, 5])
        result = sma(s, 3)
        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert not math.isnan(result.iloc[2])

    def test_correct_value(self):
        s = _series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_same_length_as_input(self):
        s = _series(range(20))
        result = sma(s, 5)
        assert len(result) == len(s)

    def test_no_lookahead(self):
        """Appending a future bar must not change past SMA values."""
        s = _series([10.0, 11.0, 12.0, 13.0, 14.0])
        result_before = sma(s, 3)

        s_ext = _series([10.0, 11.0, 12.0, 13.0, 14.0, 999.0])
        result_after = sma(s_ext, 3)

        for i in range(len(s)):
            v_before = result_before.iloc[i]
            v_after = result_after.iloc[i]
            if math.isnan(v_before):
                assert math.isnan(v_after)
            else:
                assert v_before == pytest.approx(v_after)

    def test_n1_equals_series(self):
        s = _series([3.0, 1.0, 4.0])
        result = sma(s, 1)
        pd.testing.assert_series_equal(result, s, check_names=False)


# ── RSI ────────────────────────────────────────────────────────────────────────

class TestRsi:
    def test_nan_for_insufficient_history(self):
        s = _series(list(range(1, 20)))
        result = rsi(s, 14)
        # First n rows should be NaN
        assert all(math.isnan(v) for v in result.iloc[:14])

    def test_range_0_to_100(self):
        rng = np.random.default_rng(0)
        vals = 100 + np.cumsum(rng.normal(0, 1, 100))
        s = _series(vals)
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_all_gains_gives_100(self):
        """Strictly rising prices → RSI = 100."""
        s = _series([float(i) for i in range(1, 30)])
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid > 99.0).all()

    def test_all_losses_gives_0(self):
        """Strictly falling prices → RSI ≈ 0."""
        s = _series([float(30 - i) for i in range(30)])
        result = rsi(s, 14)
        valid = result.dropna()
        assert (valid < 1.0).all()

    def test_same_length_as_input(self):
        s = _series(range(50))
        result = rsi(s, 14)
        assert len(result) == len(s)

    def test_no_lookahead(self):
        """Adding a future bar must not change any past RSI value."""
        rng = np.random.default_rng(7)
        vals = list(100 + np.cumsum(rng.normal(0, 1, 40)))
        s = _series(vals)
        result_before = rsi(s, 14)

        s_ext = _series(vals + [999.0])
        result_after = rsi(s_ext, 14)

        for i in range(len(s)):
            v_before = result_before.iloc[i]
            v_after = result_after.iloc[i]
            if math.isnan(v_before):
                assert math.isnan(v_after)
            else:
                assert v_before == pytest.approx(v_after, rel=1e-9)


# ── ATR ────────────────────────────────────────────────────────────────────────

class TestAtr:
    def test_nan_for_insufficient_history(self):
        high, low, close = _ohlc(30)
        result = atr(high, low, close, 14)
        # TR[0] is non-NaN (high-low, since prev_close is NaN and pandas max skips NaN),
        # so EWM with min_periods=14 gives NaN for the first 13 rows (indices 0..12).
        assert all(math.isnan(v) for v in result.iloc[:13])
        assert not math.isnan(result.iloc[13])

    def test_positive_values(self):
        high, low, close = _ohlc(50)
        result = atr(high, low, close, 14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_same_length_as_input(self):
        high, low, close = _ohlc(50)
        result = atr(high, low, close, 14)
        assert len(result) == len(close)

    def test_no_lookahead(self):
        """Adding a future bar must not alter any past ATR value."""
        high, low, close = _ohlc(40)
        result_before = atr(high, low, close, 14)

        # Extend by one bar
        extra_idx = pd.date_range(high.index[-1] + pd.Timedelta(days=1), periods=1, freq="B")
        high_ext = pd.concat([high, pd.Series([999.0], index=extra_idx)])
        low_ext = pd.concat([low, pd.Series([1.0], index=extra_idx)])
        close_ext = pd.concat([close, pd.Series([500.0], index=extra_idx)])
        result_after = atr(high_ext, low_ext, close_ext, 14)

        for i in range(len(close)):
            v_before = result_before.iloc[i]
            v_after = result_after.iloc[i]
            if math.isnan(v_before):
                assert math.isnan(v_after)
            else:
                assert v_before == pytest.approx(v_after, rel=1e-9)

    def test_constant_candles(self):
        """Identical OHLC every bar → ATR should equal zero (no range)."""
        n = 30
        idx = pd.date_range("2010-01-01", periods=n, freq="B")
        h = pd.Series([10.0] * n, index=idx)
        l = pd.Series([10.0] * n, index=idx)
        c = pd.Series([10.0] * n, index=idx)
        result = atr(h, l, c, 14)
        valid = result.dropna()
        assert float(valid.abs().max()) <= 1e-10
