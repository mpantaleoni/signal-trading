"""Engine B — Short-term momentum in high-activity stocks.

All signal functions are pure: they receive a slice of data up to and
including the current session (df.loc[:t_timestamp]) and return scores/
booleans. No internal state is kept here; state lives in EngineBBacktest.

No-look-ahead guarantee: callers must pass df.loc[:t_ts] (inclusive) so
that the last row is the current session close.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd


def compute_activity_score(
    df: pd.DataFrame,
    n_fast: int = 21,
    n_slow: int = 252,
) -> Optional[float]:
    """Ratio of short-term to long-term average volume.

    activity = rolling(n_fast).mean(volume) / rolling(n_slow).mean(volume)

    Returns None if *df* has insufficient history (< n_slow rows) or if
    the slow mean is zero/NaN.
    """
    if len(df) < n_slow:
        return None
    volume = df["volume"]
    fast_mean = volume.rolling(n_fast, min_periods=n_fast).mean().iloc[-1]
    slow_mean = volume.rolling(n_slow, min_periods=n_slow).mean().iloc[-1]
    if pd.isna(fast_mean) or pd.isna(slow_mean) or slow_mean == 0:
        return None
    return float(fast_mean / slow_mean)


def compute_momentum_score(
    df: pd.DataFrame,
    lookback: int = 21,
    skip: int = 2,
) -> Optional[float]:
    """Return over the past *lookback* sessions, skipping the most recent *skip*.

    momentum = (adj_close[-(skip+1)] - adj_close[-(lookback+skip)]) /
                adj_close[-(lookback+skip)]

    Returns None if *df* has fewer than lookback + skip rows or if the
    base price is zero/NaN.
    """
    required = lookback + skip
    if len(df) < required:
        return None
    adj_close = df["adj_close"]
    # Index from the end: -1 is latest (skipped), -(skip+1) is most-recent used
    end_price = float(adj_close.iloc[-(skip + 1)])
    start_price = float(adj_close.iloc[-(lookback + skip)])
    if pd.isna(start_price) or start_price == 0 or pd.isna(end_price):
        return None
    return float((end_price - start_price) / start_price)


def is_rebalance_day(d: date, calendar) -> bool:
    """Return True if *d* is the first trading session of its calendar month.

    *calendar* must support the .sessions(start, end) method.
    """
    # First calendar day of the same month
    first_of_month = date(d.year, d.month, 1)
    sessions = calendar.sessions(first_of_month, d)
    if not sessions:
        return False
    return sessions[0] == d


def select_engine_b_stocks(
    universe_dfs: dict,
    t_ts: pd.Timestamp,
    top_n: int,
    activity_fast: int = 21,
    activity_slow: int = 252,
    momentum_lookback: int = 21,
    momentum_skip: int = 2,
    min_history: int = 252,
) -> list[str]:
    """Select the top-N stocks for Engine B on date *t_ts*.

    Algorithm:
    1. For each ticker in *universe_dfs*, compute activity score on
       df.loc[:t_ts].  Keep top third by activity score.
    2. For the filtered set, compute momentum score.
    3. Return top *top_n* tickers by momentum (descending).
       Ties broken deterministically by ticker alphabetical order.

    Returns an empty list if the universe is too small.
    """
    activity_scores: list[tuple[str, float]] = []

    for ticker, df in universe_dfs.items():
        df_slice = df.loc[df.index <= t_ts]
        if len(df_slice) < min_history:
            continue
        score = compute_activity_score(df_slice, n_fast=activity_fast, n_slow=activity_slow)
        if score is not None:
            activity_scores.append((ticker, score))

    if not activity_scores:
        return []

    # Keep top third (at least 1)
    activity_scores.sort(key=lambda x: (-x[1], x[0]))
    top_third_n = max(1, len(activity_scores) // 3)
    top_third = activity_scores[:top_third_n]

    # Compute momentum for the top-third
    momentum_scores: list[tuple[str, float]] = []
    for ticker, _ in top_third:
        df = universe_dfs[ticker]
        df_slice = df.loc[df.index <= t_ts]
        mom = compute_momentum_score(df_slice, lookback=momentum_lookback, skip=momentum_skip)
        if mom is not None:
            momentum_scores.append((ticker, mom))

    if not momentum_scores:
        return []

    # Sort by momentum descending, ties broken by ticker alphabetically
    momentum_scores.sort(key=lambda x: (-x[1], x[0]))
    return [ticker for ticker, _ in momentum_scores[:top_n]]
