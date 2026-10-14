"""
S&P 500 point-in-time membership from the fja05680/sp500 constituent history.

The source CSV has one row per change event.  Each row is a full snapshot of
all S&P 500 members on that date.  Format::

    date,tickers
    1996-01-02,"MMM,AXP,T,BA,..."
    1996-01-19,"MMM,AXP,T,BA,...,NEWCO"

Primary source: https://github.com/fja05680/sp500
Vendored to data/raw/sp500_history.csv.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import io
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# GitHub raw URL for the constituent-change CSV
_GITHUB_API = "https://api.github.com/repos/fja05680/sp500/contents/"
_RAW_BASE = "https://raw.githubusercontent.com/fja05680/sp500/master/"


@dataclass
class Snapshot:
    date: date
    tickers: frozenset[str]


class SP500Universe:
    """
    Point-in-time S&P 500 membership built from full snapshots.

    ``members_on(d)`` returns the snapshot for the latest change date <= d.
    It never uses data from after d (tested in test_membership.py).
    """

    def __init__(self, snapshots: list[Snapshot]) -> None:
        if not snapshots:
            raise ValueError("Snapshot list is empty")
        self._snapshots: list[Snapshot] = sorted(snapshots, key=lambda s: s.date)
        self._dates: list[date] = [s.date for s in self._snapshots]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Path) -> "SP500Universe":
        """Load from the vendored CSV file."""
        snapshots: list[Snapshot] = []

        with open(path, newline="", encoding="utf-8") as f:
            # Try to auto-detect the format
            sample = f.read(2048)
            f.seek(0)

            # The fja05680/sp500 CSV has columns: date, tickers
            # where tickers is a comma-separated list in a single quoted field.
            try:
                df = pd.read_csv(
                    f,
                    header=0,
                    dtype=str,
                    keep_default_na=False,
                )
            except Exception as exc:
                raise ValueError(f"Could not parse constituent CSV at {path}: {exc}") from exc

        if df.empty:
            raise ValueError(f"Constituent CSV at {path} is empty")

        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]

        if "date" not in df.columns:
            raise ValueError(f"Constituent CSV has no 'date' column; found: {list(df.columns)}")

        # Find the tickers column: 'tickers' or first non-date column
        ticker_col = "tickers" if "tickers" in df.columns else [
            c for c in df.columns if c != "date"
        ][0]

        for _, row in df.iterrows():
            raw_date = str(row["date"]).strip()
            raw_tickers = str(row[ticker_col]).strip()

            try:
                d = date.fromisoformat(raw_date)
            except ValueError:
                logger.warning("Skipping row with invalid date: %r", raw_date)
                continue

            tickers = frozenset(
                t.strip() for t in raw_tickers.split(",") if t.strip()
            )
            if not tickers:
                logger.warning("Snapshot on %s has no tickers; skipping", d)
                continue

            snapshots.append(Snapshot(date=d, tickers=tickers))

        if not snapshots:
            raise ValueError(f"No valid snapshots found in {path}")

        return cls(snapshots)

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------

    def members_on(self, d: date) -> frozenset[str]:
        """
        Return the S&P 500 members as of date *d*.

        Uses the latest snapshot with ``snapshot.date <= d``.  Never peeks at
        later snapshots (no look-ahead).

        Raises ValueError for dates before the first snapshot.
        """
        idx = bisect.bisect_right(self._dates, d) - 1
        if idx < 0:
            raise ValueError(
                f"Date {d} is before the first available snapshot ({self._dates[0]})"
            )
        return self._snapshots[idx].tickers

    def membership_intervals(self) -> dict[str, list[tuple[date, Optional[date]]]]:
        """
        Per-ticker list of ``(start, end)`` intervals.

        ``end=None`` means the ticker is still in the index at the last snapshot.
        A ticker can have multiple intervals if it was removed and later re-added.
        """
        result: dict[str, list[tuple[date, Optional[date]]]] = {}
        prev_set: frozenset[str] = frozenset()

        for snap in self._snapshots:
            joined = snap.tickers - prev_set
            left = prev_set - snap.tickers

            for ticker in joined:
                result.setdefault(ticker, []).append((snap.date, None))

            for ticker in left:
                if ticker in result and result[ticker]:
                    start, _ = result[ticker][-1]
                    result[ticker][-1] = (start, snap.date)

            prev_set = snap.tickers

        return result

    def ever_members(self, start: date, end: date) -> frozenset[str]:
        """
        All tickers that were S&P 500 members on *any* day in [start, end].
        """
        result: set[str] = set()

        # Members at the start date (latest snapshot <= start)
        try:
            result.update(self.members_on(start))
        except ValueError:
            pass  # start is before first snapshot; no members at start

        # All snapshots with change dates strictly in (start, end]
        for snap in self._snapshots:
            if start < snap.date <= end:
                result.update(snap.tickers)

        return frozenset(result)

    def snapshot_dates(self) -> list[date]:
        """All snapshot dates in chronological order."""
        return list(self._dates)

    def first_snapshot_date(self) -> date:
        return self._dates[0]

    def last_snapshot_date(self) -> date:
        return self._dates[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Download helper
# ──────────────────────────────────────────────────────────────────────────────

def fetch_constituents(dest: Path) -> tuple[str, str]:
    """
    Download the S&P 500 constituent CSV from fja05680/sp500 on GitHub.

    Returns (sha256_hex, filename_on_github).
    Saves to *dest* atomically.
    """
    import json

    # 1. List repo root to find the CSV filename (it includes the date)
    req = urllib.request.Request(
        _GITHUB_API,
        headers={"User-Agent": "signal-trading/1.0", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        contents = json.loads(resp.read().decode())

    csv_files = [
        f["name"] for f in contents
        if f["name"].startswith("S&P 500 Historical") and f["name"].endswith(".csv")
    ]
    if not csv_files:
        raise RuntimeError("Could not find constituent CSV in fja05680/sp500 repo")

    filename = sorted(csv_files)[-1]  # newest if multiple exist
    raw_url = _RAW_BASE + urllib.parse.quote(filename)

    # 2. Download the file
    import urllib.parse
    req2 = urllib.request.Request(raw_url, headers={"User-Agent": "signal-trading/1.0"})
    with urllib.request.urlopen(req2, timeout=60) as resp:
        data = resp.read()

    sha256 = hashlib.sha256(data).hexdigest()

    # 3. Atomic write
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)

    logger.info("Downloaded %s → %s (sha256=%s)", filename, dest, sha256[:12])
    return sha256, filename
