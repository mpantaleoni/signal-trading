"""
Cross-check yfinance adj_close vs Stooq close on a random sample.

RISK (documented in CLAUDE.md §3): Stooq may gate automated CSV downloads.
If it does, this module logs a warning and returns an empty result rather
than aborting the run.
"""
from __future__ import annotations

import logging
import random
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from .provider import DataProvider

logger = logging.getLogger(__name__)

_SAMPLE_WINDOW_DAYS = 30  # check this many recent trading days
_TOLERANCE = 0.01  # 1 % relative difference is acceptable


def crosscheck_sample(
    primary: DataProvider,
    secondary: DataProvider,
    symbols: list[str],
    sample_n: int = 20,
    *,
    end: Optional[date] = None,
    window_days: int = _SAMPLE_WINDOW_DAYS,
    tolerance: float = _TOLERANCE,
    seed: Optional[int] = None,
) -> dict:
    """
    Compare *primary* adj_close against *secondary* close for a random sample.

    Returns a dict with::

        {
          "sample": list of symbol strings,
          "checked": int,
          "skipped": int,   # secondary returned no data
          "ok": int,
          "discrepancies": [{"symbol": .., "date": .., "primary": .., "secondary": .., "pct_diff": ..}]
          "stooq_blocked": bool,
        }
    """
    if end is None:
        end = date.today() - timedelta(days=1)
    start = end - timedelta(days=window_days * 2)  # extra buffer for weekends

    rng = random.Random(seed)
    sample = rng.sample(symbols, min(sample_n, len(symbols)))

    result = {
        "sample": sample,
        "checked": 0,
        "skipped": 0,
        "ok": 0,
        "discrepancies": [],
        "stooq_blocked": False,
    }

    for symbol in sample:
        try:
            primary_df = primary.fetch_daily(symbol, start, end)
            secondary_df = secondary.fetch_daily(symbol, start, end)
        except Exception as exc:
            logger.warning("crosscheck: fetch error for %s: %s", symbol, exc)
            result["skipped"] += 1
            continue

        if primary_df.empty or secondary_df.empty:
            result["skipped"] += 1
            if secondary_df.empty:
                logger.debug("crosscheck: secondary returned no data for %s", symbol)
                result["stooq_blocked"] = True
            continue

        result["checked"] += 1

        # Align on common dates
        common = primary_df.index.intersection(secondary_df.index)
        if len(common) == 0:
            result["skipped"] += 1
            continue

        p_adj = primary_df.loc[common, "adj_close"]
        s_adj = secondary_df.loc[common, "close"]  # stooq is already adjusted

        for ts in common:
            pv = p_adj.loc[ts]
            sv = s_adj.loc[ts]
            if pd.isna(pv) or pd.isna(sv) or sv == 0:
                continue
            pct_diff = abs(pv - sv) / sv
            if pct_diff > tolerance:
                result["discrepancies"].append(
                    {
                        "symbol": symbol,
                        "date": ts.date() if hasattr(ts, "date") else ts,
                        "primary": round(float(pv), 4),
                        "secondary": round(float(sv), 4),
                        "pct_diff": round(float(pct_diff), 4),
                    }
                )

        result["ok"] += 1

    return result
