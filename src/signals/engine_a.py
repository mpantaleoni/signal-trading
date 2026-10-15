"""Engine A — Trend-filtered RSI(2) pullback signal functions.

All signal functions are pure: they receive a slice of data up to and
including the current session and return booleans / dicts of booleans.
No internal state is kept here; state lives in BacktestEngine.

No-look-ahead guarantee: callers must pass df.loc[:t] (inclusive) so that
the last row is the current session close.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from .indicators import atr, rsi, sma


def compute_regime_ok(spy_df: pd.DataFrame) -> bool:
    """Return True if SPY close[-1] > SMA(200)[-1].

    *spy_df* must contain at least 200 rows and an 'adj_close' column.
    Returns False on insufficient data or any NaN.
    """
    if len(spy_df) < 200:
        return False
    close = spy_df["adj_close"]
    sma200 = sma(close, 200)
    last_close = close.iloc[-1]
    last_sma = sma200.iloc[-1]
    if pd.isna(last_close) or pd.isna(last_sma):
        return False
    return bool(last_close > last_sma)


def compute_entry_signal(
    df: pd.DataFrame,
    rsi_entry: float,
    min_adv_usd: float,
) -> bool:
    """Return True when all Engine A entry conditions are met.

    Conditions (all must be True at the last row of *df*):
    - len(df) >= 200
    - adj_close[-1] is not NaN
    - Liquidity: 20-day avg(close * volume) >= min_adv_usd
    - adj_close[-1] > SMA(200)[-1]
    - RSI(2)[-1] < rsi_entry

    Regime filter (SPY) and membership check are done by the engine.
    """
    if len(df) < 200:
        return False

    adj_close = df["adj_close"]
    close = df["close"]
    volume = df["volume"]

    last_adj = adj_close.iloc[-1]
    if pd.isna(last_adj):
        return False

    # Liquidity: 20-day average dollar volume
    adv = (close * volume).rolling(20, min_periods=20).mean().iloc[-1]
    if pd.isna(adv) or adv < min_adv_usd:
        return False

    # Trend: above 200-day SMA
    sma200 = sma(adj_close, 200).iloc[-1]
    if pd.isna(sma200) or last_adj <= sma200:
        return False

    # RSI(2) oversold
    rsi2 = rsi(adj_close, 2).iloc[-1]
    if pd.isna(rsi2) or rsi2 >= rsi_entry:
        return False

    return True


def compute_exit_signals(
    df: pd.DataFrame,
    entry_date: date,
    max_hold_days: int,
    calendar,
) -> dict[str, bool]:
    """Return a dict of named exit conditions.

    Keys:
    - "sma5_exit":    adj_close[-1] > SMA(5)[-1]
    - "sma200_exit":  adj_close[-1] < SMA(200)[-1]
    - "time_stop":    sessions since entry_date >= max_hold_days

    *calendar* is a TradingCalendar instance used to count sessions.
    Last row of *df* is the current session (close already available).
    """
    adj_close = df["adj_close"]
    current_date = df.index[-1]
    if hasattr(current_date, "date"):
        current_date = current_date.date()

    # SMA5 exit
    sma5_val = sma(adj_close, 5).iloc[-1]
    last_adj = adj_close.iloc[-1]
    sma5_exit = bool(
        not pd.isna(sma5_val) and not pd.isna(last_adj) and last_adj > sma5_val
    )

    # SMA200 exit
    sma200_val = sma(adj_close, 200).iloc[-1]
    sma200_exit = bool(
        not pd.isna(sma200_val) and not pd.isna(last_adj) and last_adj < sma200_val
    )

    # Time stop: count trading sessions from entry_date up to and including current_date
    sessions_held = len(calendar.sessions(entry_date, current_date))
    time_stop = sessions_held >= max_hold_days

    return {
        "sma5_exit": sma5_exit,
        "sma200_exit": sma200_exit,
        "time_stop": time_stop,
    }
