"""Scanner state — JSON persistence of PositionBook + cash between daily runs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from ..portfolio.positions import OrderSide, PendingOrder, Position, PositionBook


@dataclass
class ScannerState:
    """All data that must survive between scanner runs."""

    as_of_date: Optional[date]               # last session fully processed; None = fresh start
    cash: float
    peak_equity: float
    positions: list[dict]                    # serialised Position objects
    pending_orders: list[dict]               # serialised PendingOrder objects; filled next open
    equity_history: list[list]               # [[date_str, equity], …] — last 252 entries
    trades: list[dict]                       # closed trade records (accumulated)

    # ── factories ─────────────────────────────────────────────────────────────

    @classmethod
    def fresh(cls, initial_capital: float) -> "ScannerState":
        return cls(
            as_of_date=None,
            cash=initial_capital,
            peak_equity=initial_capital,
            positions=[],
            pending_orders=[],
            equity_history=[],
            trades=[],
        )

    @classmethod
    def load(cls, path: Path, initial_capital: float) -> "ScannerState":
        if not path.exists():
            return cls.fresh(initial_capital)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            as_of_date=date.fromisoformat(data["as_of_date"]) if data.get("as_of_date") else None,
            cash=float(data["cash"]),
            peak_equity=float(data["peak_equity"]),
            positions=list(data.get("positions") or []),
            pending_orders=list(data.get("pending_orders") or []),
            equity_history=list(data.get("equity_history") or []),
            trades=list(data.get("trades") or []),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None,
            "cash": self.cash,
            "peak_equity": self.peak_equity,
            "positions": self.positions,
            "pending_orders": self.pending_orders,
            "equity_history": self.equity_history,
            "trades": self.trades,
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    # ── PositionBook round-trip ────────────────────────────────────────────────

    def to_book(self) -> PositionBook:
        """Reconstruct a live PositionBook from serialised state."""
        book = PositionBook()
        for p in self.positions:
            pos = Position(
                ticker=p["ticker"],
                shares=int(p["shares"]),
                entry_date=date.fromisoformat(p["entry_date"]),
                entry_price=float(p["entry_price"]),
                entry_cost_usd=float(p["entry_cost_usd"]),
                engine=p["engine"],
                sector=p.get("sector", "Unknown"),
            )
            book.open_position(pos)
        for o in self.pending_orders:
            book.add_pending(PendingOrder(
                ticker=o["ticker"],
                side=OrderSide(o["side"]),
                shares=int(o["shares"]),
                engine=o["engine"],
                reason=o.get("reason", ""),
            ))
        return book

    def sync_from_book(self, book: PositionBook) -> None:
        """Overwrite positions and pending_orders from a live PositionBook."""
        self.positions = [
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "entry_date": p.entry_date.isoformat(),
                "entry_price": p.entry_price,
                "entry_cost_usd": p.entry_cost_usd,
                "engine": p.engine,
                "sector": p.sector,
            }
            for p in book.open_positions()
        ]
        self.pending_orders = [
            {
                "ticker": o.ticker,
                "side": o.side.value,
                "shares": o.shares,
                "engine": o.engine,
                "reason": o.reason,
            }
            for o in book.pending_orders()
        ]

    def append_equity(self, d: date, equity: float, max_history: int = 252) -> None:
        self.equity_history.append([d.isoformat(), equity])
        if len(self.equity_history) > max_history:
            self.equity_history = self.equity_history[-max_history:]
