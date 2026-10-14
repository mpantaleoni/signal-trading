"""Fake DataProvider for offline tests."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.data.provider import CANONICAL_COLUMNS, empty_frame


def _make_bar(d: date, base: float = 100.0, volume: float = 1_000_000.0) -> dict:
    return {
        "open": base,
        "high": base * 1.01,
        "low": base * 0.99,
        "close": base,
        "adj_close": base,
        "volume": volume,
        "dividends": 0.0,
        "splits": 0.0,
    }


class FakeProvider:
    """
    A DataProvider that returns synthetic daily bars.

    Call counts are tracked so tests can assert that the cache doesn't
    make unnecessary network calls.
    """

    name: str = "fake"

    def __init__(
        self,
        symbols: list[str] | None = None,
        *,
        base_price: float = 100.0,
        price_step: float = 0.0,   # daily price increment
    ) -> None:
        self._symbols: set[str] = set(symbols or [])
        self.base_price = base_price
        self.price_step = price_step
        self.call_count: dict[str, int] = {}

    def add_symbol(self, symbol: str) -> None:
        self._symbols.add(symbol)

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        self.call_count[symbol] = self.call_count.get(symbol, 0) + 1

        if symbol not in self._symbols:
            return empty_frame()

        # Generate Mon–Fri bars (no holiday exclusions — sufficient for tests)
        bars = []
        d = start
        price = self.base_price
        while d <= end:
            if d.weekday() < 5:  # Mon–Fri
                bars.append({"date": d, **_make_bar(d, price)})
                price += self.price_step
            d += timedelta(days=1)

        if not bars:
            return empty_frame()

        df = pd.DataFrame(bars).set_index("date")
        df.index = pd.DatetimeIndex(df.index)
        return df[CANONICAL_COLUMNS]
