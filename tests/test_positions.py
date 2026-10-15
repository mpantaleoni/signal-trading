"""Tests for PositionBook state transitions."""
from __future__ import annotations

from datetime import date

import pytest

from src.portfolio.positions import OrderSide, PendingOrder, Position, PositionBook


def _pos(ticker="AAPL", shares=100, engine="A", sector="Technology"):
    return Position(
        ticker=ticker,
        shares=shares,
        entry_date=date(2010, 1, 4),
        entry_price=50.0,
        entry_cost_usd=15.0,
        engine=engine,
        sector=sector,
    )


class TestPositionBook:
    def test_open_and_get(self):
        book = PositionBook()
        p = _pos("AAPL")
        book.open_position(p)
        assert book.get("AAPL") is p

    def test_open_duplicate_raises(self):
        book = PositionBook()
        book.open_position(_pos("AAPL"))
        with pytest.raises(ValueError):
            book.open_position(_pos("AAPL"))

    def test_close_removes_position(self):
        book = PositionBook()
        book.open_position(_pos("AAPL"))
        closed = book.close_position("AAPL")
        assert closed.ticker == "AAPL"
        assert book.get("AAPL") is None

    def test_close_unknown_raises(self):
        book = PositionBook()
        with pytest.raises(KeyError):
            book.close_position("MSFT")

    def test_open_tickers(self):
        book = PositionBook()
        book.open_position(_pos("AAPL"))
        book.open_position(_pos("MSFT"))
        assert book.open_tickers == {"AAPL", "MSFT"}

    def test_count_engine(self):
        book = PositionBook()
        book.open_position(_pos("AAPL", engine="A"))
        book.open_position(_pos("MSFT", engine="A"))
        book.open_position(_pos("GOOGL", engine="B"))
        assert book.count_engine("A") == 2
        assert book.count_engine("B") == 1

    def test_count_sector(self):
        book = PositionBook()
        book.open_position(_pos("AAPL", sector="Technology"))
        book.open_position(_pos("MSFT", sector="Technology"))
        book.open_position(_pos("JPM", sector="Financials"))
        assert book.count_sector("Technology") == 2
        assert book.count_sector("Financials") == 1

    def test_total_open(self):
        book = PositionBook()
        assert book.total_open() == 0
        book.open_position(_pos("AAPL"))
        assert book.total_open() == 1
        book.open_position(_pos("MSFT"))
        assert book.total_open() == 2
        book.close_position("AAPL")
        assert book.total_open() == 1

    def test_add_and_flush_pending(self):
        book = PositionBook()
        o1 = PendingOrder("AAPL", OrderSide.BUY, 100, "A", "entry")
        o2 = PendingOrder("MSFT", OrderSide.SELL, 50, "A", "sma5_exit")
        book.add_pending(o1)
        book.add_pending(o2)
        orders = book.flush_pending()
        assert len(orders) == 2
        assert book.pending_orders() == []

    def test_flush_clears_pending(self):
        book = PositionBook()
        book.add_pending(PendingOrder("X", OrderSide.BUY, 10, "A"))
        book.flush_pending()
        assert book.pending_orders() == []

    def test_mark_to_market(self):
        book = PositionBook()
        book.open_position(_pos("AAPL", shares=100))  # entry_price=50
        book.open_position(_pos("MSFT", shares=200))  # entry_price=50
        prices = {"AAPL": 60.0, "MSFT": 40.0}
        mtm = book.mark_to_market(prices)
        assert mtm == pytest.approx(100 * 60.0 + 200 * 40.0)

    def test_mark_to_market_missing_price_uses_entry(self):
        book = PositionBook()
        book.open_position(_pos("AAPL", shares=100))  # entry_price=50
        mtm = book.mark_to_market({})  # no prices → use entry
        assert mtm == pytest.approx(100 * 50.0)
