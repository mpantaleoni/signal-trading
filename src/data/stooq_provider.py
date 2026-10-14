"""Stooq DataProvider — stdlib urllib only, used for price-level cross-checks."""
from __future__ import annotations

import csv
import io
import logging
import urllib.error
import urllib.request
from datetime import date, timedelta

import pandas as pd

from .provider import DataProvider, empty_frame

logger = logging.getLogger(__name__)

_BASE_URL = "https://stooq.com/q/d/l/"
_TIMEOUT = 15  # seconds


def _stooq_symbol(symbol: str) -> str:
    """Convert a yfinance-style symbol to stooq format (lower-case + .us)."""
    # BRK-B → brk-b.us, AAPL → aapl.us
    return symbol.lower().replace(".", "-") + ".us"


class StooqProvider:
    """
    Fetch adjusted daily OHLCV from stooq.com.

    Only close (and open/high/low/volume) is available — no dividend or split
    data — so this provider is used for price-level cross-checks only.

    RISK: Stooq may gate automated CSV downloads.  If blocked, this provider
    returns empty_frame() with a WARNING log rather than raising, so the
    cross-check degrades gracefully to a stub.
    """

    name: str = "stooq"

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        stooq_sym = _stooq_symbol(symbol)
        url = (
            f"{_BASE_URL}?s={stooq_sym}&i=d"
            f"&d1={start.strftime('%Y%m%d')}"
            f"&d2={end.strftime('%Y%m%d')}"
        )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                text = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                logger.warning("Stooq blocked download for %s (HTTP %s)", symbol, exc.code)
                return empty_frame()
            raise RuntimeError(f"Stooq HTTP error for {symbol}: {exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Stooq network error for {symbol}: {exc}") from exc

        # Stooq returns a plain CSV; first row may be "No data found" if blocked
        if not text.strip() or "No data" in text[:50]:
            logger.debug("Stooq: no data for %s", symbol)
            return empty_frame()

        try:
            raw = pd.read_csv(io.StringIO(text), parse_dates=["Date"])
        except Exception as exc:
            logger.warning("Stooq: could not parse CSV for %s: %s", symbol, exc)
            return empty_frame()

        if raw.empty or "Date" not in raw.columns:
            return empty_frame()

        raw.columns = [c.lower() for c in raw.columns]
        raw = raw.rename(columns={"date": "index_col"})
        raw = raw.set_index("index_col")
        raw.index.name = None

        # Stooq provides: open, high, low, close, volume — all adjusted
        # Map to canonical; adj_close = close (stooq is adjusted); dividends/splits = 0
        rename = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        df = pd.DataFrame(index=raw.index)
        for src, dst in rename.items():
            df[dst] = raw[src] if src in raw.columns else float("nan")
        df["adj_close"] = df["close"]
        df["dividends"] = 0.0
        df["splits"] = 0.0

        # Ensure tz-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df = df.sort_index()

        return df[["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]]
