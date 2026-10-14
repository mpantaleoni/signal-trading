"""Trading calendar derived from SPY daily bars (no extra dependency)."""
from __future__ import annotations

import bisect
from datetime import date, timedelta
from typing import Optional

import pandas as pd


class TradingCalendar:
    """
    An ordered set of trading session dates built from a known data series
    (typically SPY).  No external holiday-calendar library is used.
    """

    def __init__(self, session_dates: list[date]) -> None:
        self._dates: list[date] = sorted(set(session_dates))

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_spy_data(cls, df: pd.DataFrame) -> "TradingCalendar":
        """Build from the DatetimeIndex of a SPY (or any liquid ETF) DataFrame."""
        dates = [
            idx.date() if hasattr(idx, "date") else idx
            for idx in df.index
        ]
        return cls(dates)

    @classmethod
    def from_dates(cls, dates: list[date]) -> "TradingCalendar":
        """Build from an explicit list of dates (for tests)."""
        return cls(dates)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def sessions(self, start: date, end: date) -> list[date]:
        """All trading sessions in [start, end], chronological."""
        lo = bisect.bisect_left(self._dates, start)
        hi = bisect.bisect_right(self._dates, end)
        return self._dates[lo:hi]

    def is_session(self, d: date) -> bool:
        """True if *d* is a known trading session."""
        idx = bisect.bisect_left(self._dates, d)
        return idx < len(self._dates) and self._dates[idx] == d

    def last_completed_session(self, as_of: Optional[date] = None) -> date:
        """
        Most recent session strictly before *as_of* (default: today).
        Raises ValueError if the calendar has no sessions before that date.
        """
        cutoff = as_of if as_of is not None else date.today()
        idx = bisect.bisect_left(self._dates, cutoff) - 1
        if idx < 0:
            raise ValueError(
                f"No completed sessions in calendar before {cutoff}"
            )
        return self._dates[idx]

    def prev_sessions(self, d: date, n: int) -> list[date]:
        """
        Last *n* sessions strictly before *d*, in chronological order.
        Returns fewer than *n* if the calendar doesn't have enough.
        """
        idx = bisect.bisect_left(self._dates, d)
        start = max(0, idx - n)
        return self._dates[start:idx]

    def next_session(self, d: date) -> Optional[date]:
        """First session strictly after *d*, or None if calendar ends."""
        idx = bisect.bisect_right(self._dates, d)
        return self._dates[idx] if idx < len(self._dates) else None

    def at_or_before(self, d: date) -> Optional[date]:
        """Latest session <= *d*, or None."""
        idx = bisect.bisect_right(self._dates, d) - 1
        return self._dates[idx] if idx >= 0 else None

    def __len__(self) -> int:
        return len(self._dates)
