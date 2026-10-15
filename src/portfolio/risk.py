"""Risk layer: SectorMap and RiskManager."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .positions import PositionBook


@dataclass
class SectorMap:
    """Ticker → GICS sector mapping."""
    _data: dict[str, str]

    @classmethod
    def from_csv(cls, path: Path) -> "SectorMap":
        """Load from a two-column CSV: ticker,sector."""
        data: dict[str, str] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("ticker", "").strip().upper()
                sector = row.get("sector", "Unknown").strip()
                if ticker:
                    data[ticker] = sector
        return cls(data)

    @classmethod
    def empty(cls) -> "SectorMap":
        return cls({})

    def get(self, ticker: str) -> str:
        return self._data.get(ticker.upper(), "Unknown")


class RiskManager:
    """Enforces position limits and drawdown rules.

    Drawdown is measured from the highest equity peak seen since construction.
    """

    def __init__(
        self,
        max_positions_engine_a: int,
        max_positions_combined: int,
        sector_cap: int,
        drawdown_warning_pct: float,
        drawdown_stop_pct: float,
        initial_equity: float,
        sector_map: Optional[SectorMap] = None,
        max_positions_engine_b: int = 10,
    ) -> None:
        self._max_engine_a = max_positions_engine_a
        self._max_engine_b = max_positions_engine_b
        self._max_combined = max_positions_combined
        self._sector_cap = sector_cap
        self._warn_pct = drawdown_warning_pct
        self._stop_pct = drawdown_stop_pct
        self._peak_equity = initial_equity
        self._sector_map = sector_map or SectorMap.empty()
        self._stopped = False

    # ── equity / drawdown ─────────────────────────────────────────────────────

    def update_equity(self, equity: float) -> None:
        """Update peak equity and check drawdown thresholds."""
        if equity > self._peak_equity:
            self._peak_equity = equity
            self._stopped = False  # recover once new peak reached

    @property
    def current_drawdown(self) -> float:
        """Fraction below peak (0.0 = at peak, 0.20 = 20% below peak)."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - (self._peak_equity / self._peak_equity))

    def drawdown_from_peak(self, equity: float) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - equity / self._peak_equity)

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    def check_drawdown(self, equity: float) -> str:
        """Check drawdown status. Returns 'ok', 'warning', or 'stopped'."""
        dd = self.drawdown_from_peak(equity)
        if dd >= self._stop_pct:
            self._stopped = True
            return "stopped"
        if dd >= self._warn_pct:
            return "warning"
        return "ok"

    # ── position limit checks ─────────────────────────────────────────────────

    def can_add_position(
        self,
        book: PositionBook,
        engine: str,
        ticker: str,
    ) -> bool:
        """Return True if adding a position is allowed by all risk rules."""
        if self._stopped:
            return False

        # Combined limit
        if book.total_open() >= self._max_combined:
            return False

        # Engine A limit
        if engine == "A" and book.count_engine("A") >= self._max_engine_a:
            return False

        # Engine B limit
        if engine == "B" and book.count_engine("B") >= self._max_engine_b:
            return False

        # Sector cap
        sector = self._sector_map.get(ticker)
        if book.count_sector(sector) >= self._sector_cap:
            return False

        return True

    def sector_of(self, ticker: str) -> str:
        return self._sector_map.get(ticker)
