"""Position tracking: Position, PendingOrder, PositionBook."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Position:
    """An open equity position."""
    ticker: str
    shares: int
    entry_date: date
    entry_price: float        # fill price (open of t+1)
    entry_cost_usd: float     # commission + slippage paid on entry
    engine: str               # "A" | "B"
    sector: str = "Unknown"

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price + self.entry_cost_usd


@dataclass
class PendingOrder:
    """An order waiting to be filled at the next open."""
    ticker: str
    side: OrderSide
    shares: int               # 0 shares → skip
    engine: str
    reason: str = ""          # e.g. "sma5_exit", "time_stop", "entry"


@dataclass
class PositionBook:
    """Tracks open positions and pending orders."""

    _positions: dict[str, Position] = field(default_factory=dict)
    _pending: list[PendingOrder] = field(default_factory=list)

    # ── read-only queries ──────────────────────────────────────────────────────

    @property
    def open_tickers(self) -> set[str]:
        return set(self._positions.keys())

    def get(self, ticker: str) -> Optional[Position]:
        return self._positions.get(ticker)

    def open_positions(self) -> list[Position]:
        return list(self._positions.values())

    def pending_orders(self) -> list[PendingOrder]:
        return list(self._pending)

    def count_engine(self, engine: str) -> int:
        return sum(1 for p in self._positions.values() if p.engine == engine)

    def count_sector(self, sector: str) -> int:
        return sum(1 for p in self._positions.values() if p.sector == sector)

    def total_open(self) -> int:
        return len(self._positions)

    # ── mutations ─────────────────────────────────────────────────────────────

    def add_pending(self, order: PendingOrder) -> None:
        self._pending.append(order)

    def flush_pending(self) -> list[PendingOrder]:
        """Return and clear all pending orders."""
        orders = self._pending[:]
        self._pending.clear()
        return orders

    def open_position(self, position: Position) -> None:
        if position.ticker in self._positions:
            raise ValueError(f"Position already open for {position.ticker}")
        self._positions[position.ticker] = position

    def close_position(self, ticker: str) -> Position:
        if ticker not in self._positions:
            raise KeyError(f"No open position for {ticker}")
        return self._positions.pop(ticker)

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Return total market value of all open positions."""
        return sum(
            p.shares * prices.get(p.ticker, p.entry_price)
            for p in self._positions.values()
        )
