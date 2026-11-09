"""Cross-sectional momentum signals.

Signals
-------
compute_momentum_12_1  : trailing 12-1 return (skip most-recent month)
compute_52wh_pct       : price as fraction of 52-week high
compute_spy_regime     : True iff SPY 12-1 momentum > 0 AND close > 200-day SMA
compute_stock_above_sma: True iff stock adj_close > its N-day SMA (stock trend filter)
select_momentum_stocks : composite rank → top_n tickers (long leg)
select_short_stocks    : composite rank → bottom short_n tickers (short leg)
is_rebalance_day       : True iff d is the first trading session of its month

No look-ahead: every function accepts df.loc[:t_ts] slices only.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


def compute_momentum_12_1(df: pd.DataFrame) -> Optional[float]:
    """12-1 month momentum: return from 252 sessions ago to 21 sessions ago.

    Uses adjusted close.  Returns None if fewer than 273 rows are available.

    Indices: df must be sliced to the signal date (df.loc[:t_ts]).
    """
    if len(df) < 273:
        return None
    end_price = float(df["adj_close"].iloc[-21])
    start_price = float(df["adj_close"].iloc[-273])
    if start_price <= 0:
        return None
    return (end_price - start_price) / start_price


def compute_52wh_pct(df: pd.DataFrame) -> Optional[float]:
    """52-week-high percentage: close[t] / max(high[t-252:t]).

    Returns None if fewer than 252 rows are available.
    """
    if len(df) < 252:
        return None
    high_52w = float(df["high"].iloc[-252:].max())
    if high_52w <= 0:
        return None
    close_t = float(df["adj_close"].iloc[-1])
    return close_t / high_52w


def compute_spy_regime(spy_df: pd.DataFrame) -> bool:
    """Return True iff SPY passes BOTH regime filters (invest), False → cash.

    Filters (both must be true):
      1. SPY 12-1 momentum > 0
      2. SPY adj_close > 200-day SMA (trend confirmation)

    spy_df must be sliced to the signal date (spy_df.loc[:t_ts]).
    If either condition cannot be computed (insufficient history), return False.
    """
    score = compute_momentum_12_1(spy_df)
    if score is None or score <= 0:
        return False
    # 200-day SMA — we already know len >= 273 (momentum passed), so >= 200 guaranteed
    sma200 = float(spy_df["adj_close"].iloc[-200:].mean())
    close_t = float(spy_df["adj_close"].iloc[-1])
    return bool(close_t > sma200)


def compute_stock_above_sma(df: pd.DataFrame, sma_period: int = 200) -> bool:
    """Return True iff adj_close[-1] > mean(adj_close[-sma_period:]).

    Used as a stock-level trend filter: only include stocks in an uptrend.
    Returns False if there are fewer than sma_period rows.

    df must be sliced to the signal date (df.loc[:t_ts]).
    """
    if len(df) < sma_period:
        return False
    sma = float(df["adj_close"].iloc[-sma_period:].mean())
    return bool(float(df["adj_close"].iloc[-1]) > sma)


def is_rebalance_day(d: date, calendar) -> bool:
    """Return True iff d is the first trading session of its month.

    calendar must expose a .sessions(start, end) method.
    """
    first_of_month = d.replace(day=1)
    sessions = calendar.sessions(first_of_month, d)
    return bool(sessions and sessions[0] == d)


def select_momentum_stocks(
    universe_dfs: dict[str, pd.DataFrame],
    t_ts: pd.Timestamp,
    top_n: int,
) -> list[str]:
    """Rank universe by composite momentum score and return top_n tickers.

    Steps:
    1. Slice each df to t_ts (no look-ahead).
    2. Compute 12-1 momentum and 52-week-high pct for every ticker.
    3. Drop tickers where either score is None.
    4. Rank each score 0→1 (percentile rank, method='average').
    5. Composite = 0.5 * rank_12_1 + 0.5 * rank_52wh.
    6. Return top_n descending; alphabetical tie-break.

    Parameters
    ----------
    universe_dfs : dict of ticker → full DataFrame (will be sliced internally)
    t_ts         : signal timestamp (inclusive upper bound for slicing)
    top_n        : number of stocks to select
    """
    scores: dict[str, tuple[float, float]] = {}

    for ticker, df in universe_dfs.items():
        df_slice = df.loc[df.index <= t_ts]
        m12 = compute_momentum_12_1(df_slice)
        w52 = compute_52wh_pct(df_slice)
        if m12 is None or w52 is None:
            continue
        scores[ticker] = (m12, w52)

    if not scores:
        return []

    tickers = sorted(scores.keys())  # alphabetical for stable sort
    m12_vals = pd.Series([scores[t][0] for t in tickers], index=tickers)
    w52_vals = pd.Series([scores[t][1] for t in tickers], index=tickers)

    rank_m12 = m12_vals.rank(pct=True, method="average")
    rank_w52 = w52_vals.rank(pct=True, method="average")
    composite = 0.5 * rank_m12 + 0.5 * rank_w52

    # Sort descending by composite; alphabetical tie-break already baked in
    # because tickers list was alphabetically sorted before ranking
    ranked = composite.sort_values(ascending=False)
    return list(ranked.index[:top_n])


def select_short_stocks(
    universe_dfs: dict[str, pd.DataFrame],
    t_ts: pd.Timestamp,
    short_n: int,
) -> list[str]:
    """Rank universe by composite momentum score and return bottom short_n tickers.

    Mirror of select_momentum_stocks — same composite score, ascending sort.
    Tickers with insufficient history (None scores) are excluded.

    Parameters
    ----------
    universe_dfs : dict of ticker → full DataFrame (will be sliced internally)
    t_ts         : signal timestamp (inclusive upper bound for slicing)
    short_n      : number of stocks to short
    """
    scores: dict[str, tuple[float, float]] = {}

    for ticker, df in universe_dfs.items():
        df_slice = df.loc[df.index <= t_ts]
        m12 = compute_momentum_12_1(df_slice)
        w52 = compute_52wh_pct(df_slice)
        if m12 is None or w52 is None:
            continue
        scores[ticker] = (m12, w52)

    if not scores:
        return []

    tickers = sorted(scores.keys())  # alphabetical for stable sort
    m12_vals = pd.Series([scores[t][0] for t in tickers], index=tickers)
    w52_vals = pd.Series([scores[t][1] for t in tickers], index=tickers)

    rank_m12 = m12_vals.rank(pct=True, method="average")
    rank_w52 = w52_vals.rank(pct=True, method="average")
    composite = 0.5 * rank_m12 + 0.5 * rank_w52

    # Sort ascending (worst momentum first) for the short leg
    ranked = composite.sort_values(ascending=True)
    return list(ranked.index[:short_n])
