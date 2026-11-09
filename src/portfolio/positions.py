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
    """An open equity position (long or short)."""
    ticker: str
    shares: int
    entry_date: date
    entry_price: float        # fill price (open of t+1)
    entry_cost_usd: float     # commission + slippage paid on entry
    engine: str               # "A" | "B"
    sector: str = "Unknown"
    direction: str = "long"   # "long" or "short"

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
    reason: str = ""          # e.g. "rebalance_exit", "regime_exit", "entry"
    direction: str = "long"   # "long" or "short"


@dataclass
class PositionBook:
    """Tracks open long and short positions plus pending orders.

    Long positions are stored in _positions (keyed by ticker).
    Short positions are stored in _short_positions (keyed by ticker).
    A ticker cannot appear in both books simultaneously (momentum rank ensures
    top-N and bottom-N are disjoint, but the book enforces this via open_position).

    Mark-to-market semantics:
        equity = cash + mark_to_market(prices)
    where mark_to_market returns  Σ(long_shares × price) − Σ(short_shares × price).
    Cash includes short-sale proceeds received at entry, so subtracting the
    current short notional correctly captures short P&L.
    """

    _positions: dict[str, Position] = field(default_factory=dict)
    _short_positions: dict[str, Position] = field(default_factory=dict)
    _pending: list[PendingOrder] = field(default_factory=list)

    # ── read-only queries ──────────────────────────────────────────────────────

    @property
    def open_tickers(self) -> set[str]:
        """Long position tickers."""
        return set(self._positions.keys())

    @property
    def short_tickers(self) -> set[str]:
        """Short position tickers."""
        return set(self._short_positions.keys())

    def get(self, ticker: str) -> Optional[Position]:
        """Return the long position for ticker, or None."""
        return self._positions.get(ticker)

    def get_short(self, ticker: str) -> Optional[Position]:
        """Return the short position for ticker, or None."""
        return self._short_positions.get(ticker)

    def open_positions(self) -> list[Position]:
        """All open positions (long and short)."""
        return list(self._positions.values()) + list(self._short_positions.values())

    def long_positions(self) -> list[Position]:
        return list(self._positions.values())

    def short_positions(self) -> list[Position]:
        return list(self._short_positions.values())

    def pending_orders(self) -> list[PendingOrder]:
        return list(self._pending)

    def count_engine(self, engine: str) -> int:
        return sum(1 for p in self.open_positions() if p.engine == engine)

    def count_sector(self, sector: str) -> int:
        """Count combined long + short positions in the given sector."""
        return sum(1 for p in self.open_positions() if p.sector == sector)

    def total_open(self) -> int:
        """Total open positions, long + short combined."""
        return len(self._positions) + len(self._short_positions)

    # ── mutations ─────────────────────────────────────────────────────────────

    def add_pending(self, order: PendingOrder) -> None:
        self._pending.append(order)

    def flush_pending(self) -> list[PendingOrder]:
        """Return and clear all pending orders."""
        orders = self._pending[:]
        self._pending.clear()
        return orders

    def open_position(self, position: Position) -> None:
        """Open a long position. Raises if ticker already has a long position."""
        if position.ticker in self._positions:
            raise ValueError(f"Long position already open for {position.ticker}")
        self._positions[position.ticker] = position

    def close_position(self, ticker: str) -> Position:
        """Close and return the long position. Raises if not found."""
        if ticker not in self._positions:
            raise KeyError(f"No open long position for {ticker}")
        return self._positions.pop(ticker)

    def open_short_position(self, position: Position) -> None:
        """Open a short position. Raises if ticker already has a short position."""
        if position.ticker in self._short_positions:
            raise ValueError(f"Short position already open for {position.ticker}")
        self._short_positions[position.ticker] = position

    def close_short_position(self, ticker: str) -> Position:
        """Close and return the short position. Raises if not found."""
        if ticker not in self._short_positions:
            raise KeyError(f"No open short position for {ticker}")
        return self._short_positions.pop(ticker)

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Return net market value contribution of all open positions.

        Long positions contribute positively; short positions negatively.
        Adding this value to cash gives total equity.
        """
        long_val = sum(
            p.shares * prices.get(p.ticker, p.entry_price)
            for p in self._positions.values()
        )
        short_val = sum(
            p.shares * prices.get(p.ticker, p.entry_price)
            for p in self._short_positions.values()
        )
        return long_val - short_val
