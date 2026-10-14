"""
Tests for src/data/yfinance_provider.py and src/data/stooq_provider.py.

Network tests are marked @pytest.mark.network and skipped by default.
Offline tests validate schema conversion on synthetic / recorded responses.
"""
from __future__ import annotations

from datetime import date
from io import StringIO

import pandas as pd
import pytest

from src.data.provider import CANONICAL_COLUMNS, validate_schema
from tests.fakes.fake_provider import FakeProvider


# ──────────────────────────────────────────────────────────────────────────────
# FakeProvider (used in many other tests, validated here)
# ──────────────────────────────────────────────────────────────────────────────

class TestFakeProvider:
    def test_returns_canonical_schema(self):
        p = FakeProvider(["AAPL"])
        df = p.fetch_daily("AAPL", date(2022, 1, 3), date(2022, 1, 31))
        validate_schema(df)

    def test_returns_empty_for_unknown_symbol(self):
        p = FakeProvider(["AAPL"])
        df = p.fetch_daily("UNKNOWN", date(2022, 1, 3), date(2022, 1, 7))
        assert df.empty
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_index_is_tz_naive(self):
        p = FakeProvider(["X"])
        df = p.fetch_daily("X", date(2022, 1, 3), date(2022, 1, 7))
        assert df.index.tz is None

    def test_call_count_increments(self):
        p = FakeProvider(["X"])
        p.fetch_daily("X", date(2022, 1, 3), date(2022, 1, 7))
        p.fetch_daily("X", date(2022, 1, 3), date(2022, 1, 7))
        assert p.call_count["X"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# YFinanceProvider — offline schema tests
# ──────────────────────────────────────────────────────────────────────────────

class TestYFinanceProviderSchema:
    def _make_raw_yf(self, dates: list[date]) -> pd.DataFrame:
        """Simulate the DataFrame that yfinance returns."""
        idx = pd.DatetimeIndex(dates)
        df = pd.DataFrame(
            {
                "Open": 100.0,
                "High": 102.0,
                "Low": 98.0,
                "Close": 100.0,
                "Adj Close": 99.0,
                "Volume": 1_000_000,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            },
            index=idx,
        )
        return df

    def test_rename_produces_canonical_columns(self):
        """The renaming in YFinanceProvider maps to all CANONICAL_COLUMNS."""
        from src.data.yfinance_provider import _RENAME
        expected = set(CANONICAL_COLUMNS)
        mapped = set(_RENAME.values())
        # All canonical columns must be covered by the rename map
        # (some may be added as 0 if missing, e.g. 'splits' on ETFs)
        assert expected <= mapped | {"dividends", "splits"}

    def test_adj_close_separate_from_close(self):
        """After rename, 'close' and 'adj_close' are distinct columns."""
        from src.data.yfinance_provider import _RENAME
        assert _RENAME["Close"] == "close"
        assert _RENAME["Adj Close"] == "adj_close"
        assert _RENAME["Close"] != _RENAME["Adj Close"]


# ──────────────────────────────────────────────────────────────────────────────
# StooqProvider — offline schema / parsing tests
# ──────────────────────────────────────────────────────────────────────────────

class TestStooqProviderSchema:
    def test_stooq_symbol_conversion(self):
        from src.data.stooq_provider import _stooq_symbol
        assert _stooq_symbol("AAPL") == "aapl.us"
        assert _stooq_symbol("BRK-B") == "brk-b.us"

    def test_parse_valid_csv(self, monkeypatch):
        """Parsing a valid Stooq CSV response returns canonical schema."""
        from src.data.stooq_provider import StooqProvider
        import urllib.request

        csv_text = (
            "Date,Open,High,Low,Close,Volume\n"
            "2022-01-03,148.0,150.0,146.0,149.0,1000000\n"
            "2022-01-04,149.0,151.0,147.0,150.0,1100000\n"
        )

        class FakeResponse:
            def read(self):
                return csv_text.encode()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResponse())
        p = StooqProvider()
        df = p.fetch_daily("AAPL", date(2022, 1, 3), date(2022, 1, 4))

        assert not df.empty
        validate_schema(df)
        assert df["adj_close"].equals(df["close"])
        assert (df["dividends"] == 0.0).all()

    def test_blocked_response_returns_empty(self, monkeypatch):
        """A 403 response returns empty_frame() without raising."""
        from src.data.stooq_provider import StooqProvider
        import urllib.request, urllib.error

        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.HTTPError(None, 403, "Forbidden", {}, None))
        )
        p = StooqProvider()
        df = p.fetch_daily("AAPL", date(2022, 1, 3), date(2022, 1, 4))
        assert df.empty


# ──────────────────────────────────────────────────────────────────────────────
# Network tests (skipped by default)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.network
class TestYFinanceProviderNetwork:
    def test_fetch_spy_returns_data(self):
        from src.data.yfinance_provider import YFinanceProvider
        p = YFinanceProvider()
        df = p.fetch_daily("SPY", date(2023, 1, 2), date(2023, 1, 31))
        assert not df.empty
        validate_schema(df)
        assert (df["close"] > 0).all()
        assert df.index.tz is None

    def test_unknown_ticker_returns_empty(self):
        from src.data.yfinance_provider import YFinanceProvider
        p = YFinanceProvider()
        df = p.fetch_daily("THISSYMBOLCANNOTEXIST999", date(2023, 1, 2), date(2023, 1, 10))
        assert df.empty


@pytest.mark.network
class TestStooqProviderNetwork:
    def test_fetch_spy_or_blocked(self):
        from src.data.stooq_provider import StooqProvider
        p = StooqProvider()
        df = p.fetch_daily("SPY", date(2023, 1, 2), date(2023, 1, 31))
        # Either returns data (ok) or empty (blocked) — both acceptable
        if not df.empty:
            validate_schema(df)
