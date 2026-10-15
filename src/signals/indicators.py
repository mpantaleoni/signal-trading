"""Vectorized technical indicators — pure functions, no internal state.

All functions return a same-length Series with NaN for rows with insufficient history.
No look-ahead: all values at index i depend only on data[0..i].
"""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average over *n* periods (Wilder/standard).

    Returns NaN for the first n-1 rows.
    """
    return series.rolling(n, min_periods=n).mean()


def rsi(series: pd.Series, n: int) -> pd.Series:
    """Wilder RSI over *n* periods.

    Uses EWM with com=n-1, adjust=False to match Wilder's smoothing.
    Returns values in [0, 100]. NaN for the first n rows.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=n - 1, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(com=n - 1, adjust=False, min_periods=n).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0, RSI should be 100
    result = result.where(avg_loss != 0, 100.0)
    return result


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """Average True Range over *n* periods using Wilder EWM smoothing.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    Returns NaN for the first n rows.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=n - 1, adjust=False, min_periods=n).mean()
