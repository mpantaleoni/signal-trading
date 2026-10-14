"""yfinance DataProvider implementation."""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from .provider import CANONICAL_COLUMNS, DataProvider, empty_frame

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_MARKET_CLOSE_HOUR = 16  # 4 PM ET

# yfinance column → canonical column
_RENAME: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}


def _us_session_has_closed() -> bool:
    """True if the US equity session for today has already closed."""
    from datetime import datetime
    now_et = datetime.now(_ET)
    return now_et.hour >= _MARKET_CLOSE_HOUR


class YFinanceProvider:
    """Fetch daily OHLCV data from Yahoo Finance via yfinance."""

    name: str = "yfinance"

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        Fetch [start, end] inclusive.  Returns empty_frame() when Yahoo has
        no data for the symbol.  Raises on network or parse errors.
        """
        import yfinance as yf

        # yfinance end is exclusive, so add one day
        yf_end = end + timedelta(days=1)

        try:
            raw = yf.download(
                symbol,
                start=start.isoformat(),
                end=yf_end.isoformat(),
                auto_adjust=False,
                actions=True,
                progress=False,
                multi_level_index=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"yfinance network error for {symbol}: {exc}"
            ) from exc

        if raw is None or raw.empty:
            logger.debug("%s: no data from yfinance", symbol)
            return empty_frame()

        # Flatten MultiIndex if present (older yfinance versions)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Rename to canonical
        raw = raw.rename(columns=_RENAME)
        missing = [c for c in CANONICAL_COLUMNS if c not in raw.columns]
        if missing:
            # Some columns may be absent (e.g. splits for ETFs); fill with 0
            for col in missing:
                raw[col] = 0.0

        df = raw[CANONICAL_COLUMNS].copy()

        # Ensure tz-naive DatetimeIndex (yfinance may return tz-aware)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()

        # Drop today's bar if the US session has not yet closed
        if not df.empty:
            today_ts = pd.Timestamp(date.today())
            if df.index[-1] == today_ts and not _us_session_has_closed():
                df = df.iloc[:-1]

        df = df.sort_index()
        return df
