"""Tests for Phase 4: scanner state, Telegram formatting, status page."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.state import ScannerState
from src.portfolio.positions import OrderSide, PendingOrder, Position, PositionBook
from src.alerts.telegram import TelegramAlert, format_report


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_state(initial_capital: float = 100_000.0) -> ScannerState:
    return ScannerState(
        as_of_date=date(2024, 1, 15),
        cash=80_000.0,
        peak_equity=105_000.0,
        positions=[
            {
                "ticker": "AAPL",
                "shares": 10,
                "entry_date": "2024-01-10",
                "entry_price": 185.0,
                "entry_cost_usd": 12.95,
                "engine": "A",
                "sector": "Technology",
            }
        ],
        pending_orders=[
            {
                "ticker": "MSFT",
                "side": "buy",
                "shares": 5,
                "engine": "A",
                "reason": "entry",
            }
        ],
        equity_history=[["2024-01-14", 100_500.0], ["2024-01-15", 101_850.0]],
        trades=[],
    )


def _make_report():
    """Build a minimal ScanReport-like object for formatting tests."""
    from src.scanner.run import FillRecord, ScanReport

    fill = FillRecord(
        ticker="XOM",
        side="sell",
        shares=20,
        price=103.50,
        engine="A",
        reason="sma5_exit",
        pnl_usd=+240.0,
    )

    return ScanReport(
        as_of_date=date(2024, 1, 15),
        equity=101_850.0,
        cash=80_000.0,
        initial_capital=100_000.0,
        peak_equity=105_000.0,
        drawdown_pct=0.03,
        drawdown_status="ok",
        engine="a",
        open_positions=[
            {
                "ticker": "AAPL",
                "shares": 10,
                "unrealized_pnl": 150.0,
                "unrealized_pct": 0.81,
                "engine": "A",
                "sessions_held": 5,
            }
        ],
        pending_orders=[
            {"ticker": "MSFT", "side": "buy", "shares": 5, "engine": "A"}
        ],
        fills_today=[fill],
        trades_total=12,
    )


# ── ScannerState serialisation ─────────────────────────────────────────────────

class TestScannerState:
    def test_fresh_state(self):
        s = ScannerState.fresh(100_000.0)
        assert s.as_of_date is None
        assert s.cash == 100_000.0
        assert s.peak_equity == 100_000.0
        assert s.positions == []
        assert s.pending_orders == []
        assert s.equity_history == []
        assert s.trades == []

    def test_save_and_load_roundtrip(self):
        s = _make_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            s.save(path)
            loaded = ScannerState.load(path, initial_capital=100_000.0)

        assert loaded.as_of_date == date(2024, 1, 15)
        assert loaded.cash == 80_000.0
        assert loaded.peak_equity == 105_000.0
        assert len(loaded.positions) == 1
        assert loaded.positions[0]["ticker"] == "AAPL"
        assert len(loaded.pending_orders) == 1
        assert loaded.pending_orders[0]["ticker"] == "MSFT"
        assert len(loaded.equity_history) == 2

    def test_load_missing_file_returns_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonexistent.json"
            s = ScannerState.load(path, initial_capital=50_000.0)
        assert s.as_of_date is None
        assert s.cash == 50_000.0

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "nested" / "state.json"
            s = ScannerState.fresh(100_000.0)
            s.save(path)
            assert path.exists()

    def test_to_book_reconstructs_positions(self):
        s = _make_state()
        book = s.to_book()
        open_positions = book.open_positions()
        assert len(open_positions) == 1
        assert open_positions[0].ticker == "AAPL"
        assert open_positions[0].shares == 10
        assert open_positions[0].engine == "A"

    def test_to_book_reconstructs_pending_orders(self):
        s = _make_state()
        book = s.to_book()
        orders = book.pending_orders()
        assert len(orders) == 1
        assert orders[0].ticker == "MSFT"
        assert orders[0].side == OrderSide.BUY

    def test_sync_from_book(self):
        s = ScannerState.fresh(100_000.0)
        book = PositionBook()
        pos = Position(
            ticker="GOOG",
            shares=3,
            entry_date=date(2024, 1, 10),
            entry_price=140.0,
            entry_cost_usd=12.95,
            engine="B",
            sector="Technology",
        )
        book.open_position(pos)
        book.add_pending(PendingOrder(
            ticker="AMZN", side=OrderSide.SELL, shares=2, engine="A", reason="exit"
        ))
        s.sync_from_book(book)
        assert len(s.positions) == 1
        assert s.positions[0]["ticker"] == "GOOG"
        assert len(s.pending_orders) == 1
        assert s.pending_orders[0]["ticker"] == "AMZN"

    def test_append_equity_keeps_max_history(self):
        s = ScannerState.fresh(100_000.0)
        for i in range(300):
            s.append_equity(date(2024, 1, 1), 100_000.0 + i, max_history=252)
        assert len(s.equity_history) == 252

    def test_json_is_human_readable(self):
        """Saved JSON should be indented (not one-liner)."""
        s = _make_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            s.save(path)
            raw = path.read_text()
        assert "\n" in raw


# ── Telegram formatting ────────────────────────────────────────────────────────

class TestFormatReport:
    def test_contains_date(self):
        report = _make_report()
        text = format_report(report)
        assert "2024-01-15" in text

    def test_contains_equity(self):
        report = _make_report()
        text = format_report(report)
        assert "101,850" in text

    def test_contains_fill(self):
        report = _make_report()
        text = format_report(report)
        assert "XOM" in text
        assert "SELL" in text

    def test_contains_position(self):
        report = _make_report()
        text = format_report(report)
        assert "AAPL" in text

    def test_contains_pending_order(self):
        report = _make_report()
        text = format_report(report)
        assert "MSFT" in text

    def test_stopped_flag_shown(self):
        report = _make_report()
        report.drawdown_status = "stopped"
        text = format_report(report)
        assert "STOPPED" in text.upper()

    def test_no_fills_shows_no_fills_section(self):
        report = _make_report()
        report.fills_today = []
        text = format_report(report)
        assert "Fills" not in text or "Fills today (0)" not in text

    def test_no_orders_shows_no_orders_message(self):
        report = _make_report()
        report.pending_orders = []
        text = format_report(report)
        assert "No orders" in text


class TestTelegramAlert:
    def test_from_env_returns_none_without_vars(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert TelegramAlert.from_env() is None

    def test_from_env_returns_instance_with_vars(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat456")
        alert = TelegramAlert.from_env()
        assert alert is not None
        assert alert.token == "token123"
        assert alert.chat_id == "chat456"

    def test_send_calls_api(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr("requests.post", mock_post)

        alert = TelegramAlert(token="tok", chat_id="cid")
        result = alert.send("Hello")
        assert result is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "tok" in call_kwargs[0][0]

    def test_send_failure_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(
            "requests.post", MagicMock(side_effect=Exception("network error"))
        )
        alert = TelegramAlert(token="tok", chat_id="cid")
        # send_failure must not raise
        alert.send_failure("something went wrong")


# ── Status page ────────────────────────────────────────────────────────────────

class TestStatusPage:
    def test_generates_index_and_json(self):
        from src.scanner.status_page import generate_status_page

        report = _make_report()
        state = _make_state()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_status_page(report, state, output_dir=out)
            assert (out / "index.html").exists()
            assert (out / "status.json").exists()

    def test_status_json_fields(self):
        from src.scanner.status_page import generate_status_page

        report = _make_report()
        state = _make_state()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_status_page(report, state, output_dir=out)
            data = json.loads((out / "status.json").read_text())

        assert data["as_of_date"] == "2024-01-15"
        assert data["equity"] == pytest.approx(101_850.0)
        assert data["drawdown_status"] == "ok"
        assert len(data["fills_today"]) == 1
        assert data["fills_today"][0]["ticker"] == "XOM"

    def test_html_contains_equity(self):
        from src.scanner.status_page import generate_status_page

        report = _make_report()
        state = _make_state()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_status_page(report, state, output_dir=out)
            html = (out / "index.html").read_text()

        assert "101,850" in html

    def test_html_contains_sparkline_when_history(self):
        from src.scanner.status_page import generate_status_page

        report = _make_report()
        state = _make_state()  # has 2 history entries

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_status_page(report, state, output_dir=out)
            html = (out / "index.html").read_text()

        assert "<svg" in html
        assert "polyline" in html

    def test_stopped_banner_shown(self):
        from src.scanner.status_page import generate_status_page

        report = _make_report()
        report.drawdown_status = "stopped"
        state = _make_state()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_status_page(report, state, output_dir=out)
            html = (out / "index.html").read_text()

        assert "STOPPED" in html

    def test_html_no_sparkline_with_single_history(self):
        from src.scanner.status_page import _sparkline_svg
        # fewer than 2 points → empty string
        assert _sparkline_svg([]) == ""
        assert _sparkline_svg([["2024-01-01", 100_000.0]]) == ""

    def test_sparkline_svg_color_green_when_rising(self):
        from src.scanner.status_page import _sparkline_svg
        svg = _sparkline_svg([["2024-01-01", 100_000.0], ["2024-01-02", 101_000.0]])
        assert "#3fb950" in svg  # green

    def test_sparkline_svg_color_red_when_falling(self):
        from src.scanner.status_page import _sparkline_svg
        svg = _sparkline_svg([["2024-01-01", 101_000.0], ["2024-01-02", 100_000.0]])
        assert "#f85149" in svg  # red
