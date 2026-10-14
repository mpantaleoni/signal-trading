"""Data integrity checks.  Each check returns a list of IntegrityFailure records."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

# Split ratios that a raw close-to-close ratio might mimic
_SPLIT_RATIOS = {2.0, 3.0, 4.0, 5.0, 0.5, 1 / 3, 0.25, 0.2, 1.5, 2 / 3}
_SPLIT_TOLERANCE = 0.04  # 4% window around each ratio


@dataclass
class IntegrityFailure:
    symbol: str
    date: Optional[date]
    check: str
    detail: str


# ──────────────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────────────

def check_missing_bars(
    symbol: str,
    df: pd.DataFrame,
    calendar_sessions: set[date],
) -> list[IntegrityFailure]:
    """Sessions inside the data range that have no bar."""
    if df.empty:
        return []

    first = _to_date(df.index[0])
    last = _to_date(df.index[-1])
    expected = {d for d in calendar_sessions if first <= d <= last}
    actual = {_to_date(ts) for ts in df.index}
    missing = sorted(expected - actual)

    return [
        IntegrityFailure(symbol, d, "missing_bar", f"No bar on {d}")
        for d in missing
    ]


def check_price_validity(
    symbol: str,
    df: pd.DataFrame,
) -> list[IntegrityFailure]:
    """Zero, negative, or NaN values in price columns."""
    failures: list[IntegrityFailure] = []
    for col in ("open", "high", "low", "close", "adj_close"):
        if col not in df.columns:
            continue
        series = df[col]
        bad = series[(series.isna()) | (series <= 0)]
        for ts, val in bad.items():
            failures.append(
                IntegrityFailure(
                    symbol, _to_date(ts), "price_validity",
                    f"{col}={val!r} (zero/negative/NaN)",
                )
            )
    return failures


def check_adjustment_jumps(
    symbol: str,
    df: pd.DataFrame,
    large_return_threshold: float = 0.40,
) -> list[IntegrityFailure]:
    """
    Two sub-checks:
    1. Raw close-to-close ratio that looks like a split ratio with no split recorded.
    2. Adjusted return in a single day above `large_return_threshold`.
    """
    failures: list[IntegrityFailure] = []
    if df.empty or len(df) < 2:
        return failures

    close = df["close"] if "close" in df.columns else None
    adj_close = df["adj_close"] if "adj_close" in df.columns else None
    splits = df["splits"] if "splits" in df.columns else None

    # Sub-check 1: raw split-like jumps without a recorded split
    if close is not None and splits is not None:
        raw_ratio = close / close.shift(1)
        for i in range(1, len(df)):
            ts = df.index[i]
            ratio = raw_ratio.iloc[i]
            if pd.isna(ratio):
                continue
            split_val = splits.iloc[i]
            recorded = not (pd.isna(split_val) or split_val == 0)
            for sr in _SPLIT_RATIOS:
                if abs(ratio - sr) / sr <= _SPLIT_TOLERANCE and not recorded:
                    failures.append(
                        IntegrityFailure(
                            symbol, _to_date(ts), "split_like_jump",
                            f"raw close ratio={ratio:.4f} ≈ {sr}, no split recorded",
                        )
                    )
                    break

    # Sub-check 2: large adjusted return
    if adj_close is not None:
        adj_ret = (adj_close / adj_close.shift(1) - 1).abs()
        big = adj_ret[adj_ret > large_return_threshold].dropna()
        for ts, ret in big.items():
            failures.append(
                IntegrityFailure(
                    symbol, _to_date(ts), "large_adj_return",
                    f"|adj_return|={ret:.2%} > {large_return_threshold:.0%}",
                )
            )

    return failures


def check_stale_prices(
    symbol: str,
    df: pd.DataFrame,
    run: int = 5,
) -> list[IntegrityFailure]:
    """5+ (or `run`) consecutive identical closes."""
    failures: list[IntegrityFailure] = []
    if df.empty or "close" not in df.columns or len(df) < run:
        return failures

    close = df["close"]
    streak = 1
    for i in range(1, len(close)):
        if close.iloc[i] == close.iloc[i - 1]:
            streak += 1
            if streak == run:
                # Flag the start of the run
                start_ts = df.index[i - run + 1]
                failures.append(
                    IntegrityFailure(
                        symbol, _to_date(start_ts), "stale_prices",
                        f"{run}+ identical closes starting {_to_date(start_ts)}, "
                        f"value={close.iloc[i]}",
                    )
                )
        else:
            streak = 1

    return failures


def check_ohlc_consistency(
    symbol: str,
    df: pd.DataFrame,
) -> list[IntegrityFailure]:
    """
    D5 extra check: high >= low, high >= open, high >= close,
    low <= open, low <= close.
    """
    failures: list[IntegrityFailure] = []
    if df.empty:
        return failures

    cols = ("open", "high", "low", "close")
    if not all(c in df.columns for c in cols):
        return failures

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    checks = [
        (h < l, "high < low"),
        (h < o, "high < open"),
        (h < c, "high < close"),
        (l > o, "low > open"),
        (l > c, "low > close"),
    ]
    for mask, desc in checks:
        for ts in df.index[mask.fillna(False)]:
            row = df.loc[ts]
            failures.append(
                IntegrityFailure(
                    symbol, _to_date(ts), "ohlc_consistency",
                    f"{desc}: O={row['open']} H={row['high']} "
                    f"L={row['low']} C={row['close']}",
                )
            )

    return failures


def check_zero_volume(
    symbol: str,
    df: pd.DataFrame,
) -> list[IntegrityFailure]:
    """D5 extra check: zero-volume trading days."""
    if df.empty or "volume" not in df.columns:
        return []
    zero = df[df["volume"] == 0]
    return [
        IntegrityFailure(symbol, _to_date(ts), "zero_volume", "volume=0")
        for ts in zero.index
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Combined runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all_checks(
    symbol: str,
    df: pd.DataFrame,
    calendar_sessions: set[date],
    *,
    large_return_threshold: float = 0.40,
    stale_price_run: int = 5,
) -> list[IntegrityFailure]:
    """Run every integrity check and return all failures."""
    out: list[IntegrityFailure] = []
    out += check_missing_bars(symbol, df, calendar_sessions)
    out += check_price_validity(symbol, df)
    out += check_adjustment_jumps(symbol, df, large_return_threshold)
    out += check_stale_prices(symbol, df, stale_price_run)
    out += check_ohlc_consistency(symbol, df)
    out += check_zero_volume(symbol, df)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_date(ts) -> date:
    return ts.date() if hasattr(ts, "date") else ts
