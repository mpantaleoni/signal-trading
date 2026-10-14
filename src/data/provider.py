"""DataProvider protocol and canonical bar schema."""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

# Canonical daily bar columns (tz-naive DatetimeIndex, session dates only).
# open/high/low/close: unadjusted prices
# adj_close: split- and dividend-adjusted close
# volume: shares traded
# dividends: dividend per share on that date (0.0 if none)
# splits: split ratio on that date (0.0 if none, e.g. 2.0 = 2-for-1)
CANONICAL_COLUMNS: list[str] = [
    "open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"
]


@runtime_checkable
class DataProvider(Protocol):
    """Protocol every data provider must satisfy."""

    name: str

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        Return daily OHLCV bars for *symbol* in [start, end] (inclusive).

        - Returns a DataFrame with a tz-naive DatetimeIndex and CANONICAL_COLUMNS.
        - Returns an **empty** DataFrame (correct schema, zero rows) when the
          provider has no data for the symbol.
        - Raises an exception (never returns empty) on genuine network /
          parse errors, so callers can distinguish "no data" from failure.
        """
        ...


def empty_frame() -> pd.DataFrame:
    """Return a zero-row DataFrame with the canonical schema."""
    return pd.DataFrame(columns=CANONICAL_COLUMNS).astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "adj_close": float,
            "volume": float,
            "dividends": float,
            "splits": float,
        }
    )


def validate_schema(df: pd.DataFrame, context: str = "") -> None:
    """Raise ValueError if *df* does not conform to the canonical schema."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame {context} missing canonical columns: {missing}"
        )
    if not df.empty and df.index.tz is not None:
        raise ValueError(
            f"DataFrame {context} has tz-aware index; expected tz-naive."
        )
