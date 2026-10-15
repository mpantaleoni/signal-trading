"""Parquet + SQLite cache for daily price data."""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .calendar import TradingCalendar
from .provider import CANONICAL_COLUMNS, DataProvider, empty_frame

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol        TEXT NOT NULL,
    provider      TEXT NOT NULL,
    first_date    TEXT,
    last_date     TEXT,
    rows          INTEGER,
    fetched_at    TEXT,
    status        TEXT DEFAULT 'ok',
    adj_fingerprint TEXT,
    retry_after   TEXT,
    PRIMARY KEY (symbol, provider)
);

CREATE TABLE IF NOT EXISTS integrity_failures (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    date       TEXT,
    check_name TEXT NOT NULL,
    detail     TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    started_at          TEXT,
    finished_at         TEXT,
    provider            TEXT,
    symbols_attempted   INTEGER DEFAULT 0,
    symbols_updated     INTEGER DEFAULT 0,
    symbols_no_data     INTEGER DEFAULT 0,
    symbols_error       INTEGER DEFAULT 0,
    symbols_skipped     INTEGER DEFAULT 0
);
"""


# ──────────────────────────────────────────────────────────────────────────────
# CacheManager
# ──────────────────────────────────────────────────────────────────────────────

class CacheManager:
    """
    Manages a local Parquet + SQLite cache for daily price data.

    Directory layout::

        <cache_dir>/
          meta.sqlite
          parquet/
            <provider>/
              <SYMBOL>.parquet
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        calendar: Optional[TradingCalendar] = None,
        overlap_sessions: int = 10,
        no_data_retry_days: int = 30,
        adj_tolerance: float = 0.02,
        config_start_date: date = date(2005, 1, 1),
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.calendar = calendar
        self.overlap_sessions = overlap_sessions
        self.no_data_retry_days = no_data_retry_days
        self.adj_tolerance = adj_tolerance
        self.config_start_date = config_start_date

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "meta.sqlite"
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_calendar(self, cal: TradingCalendar) -> None:
        self.calendar = cal

    def update_symbol(self, provider: DataProvider, symbol: str) -> str:
        """
        Incrementally update the cache for one symbol.

        Returns one of: 'skipped' | 'updated' | 'refetched' | 'no_data' | 'error'
        """
        meta = self._get_meta(provider.name, symbol)

        # ── 1. Respect no_data retry_after ──────────────────────────────────
        if meta and meta["status"] == "no_data" and meta["retry_after"]:
            retry_after = date.fromisoformat(meta["retry_after"])
            if date.today() < retry_after:
                logger.debug("%s: no_data cooldown until %s", symbol, retry_after)
                return "skipped"

        # ── 2. Check freshness ───────────────────────────────────────────────
        if meta and meta["status"] == "ok" and meta["last_date"]:
            last_date = date.fromisoformat(meta["last_date"])
            last_completed = self._last_completed_session()
            if last_date >= last_completed:
                return "skipped"

        # ── 3. Determine fetch range ─────────────────────────────────────────
        fetch_start = self.config_start_date
        full_fetch = True

        if meta and meta["status"] == "ok" and meta["last_date"]:
            last_date = date.fromisoformat(meta["last_date"])
            if self.calendar:
                prev = self.calendar.prev_sessions(last_date, self.overlap_sessions)
                if prev:
                    fetch_start = prev[0]
                    full_fetch = False
            else:
                fetch_start = last_date - timedelta(days=self.overlap_sessions * 2)
                full_fetch = False

        # ── 4. Fetch ─────────────────────────────────────────────────────────
        fetch_end = self._last_completed_session()
        try:
            new_df = provider.fetch_daily(symbol, fetch_start, fetch_end)
        except Exception as exc:
            logger.error("Fetch error for %s: %s", symbol, exc)
            self._upsert_meta(provider.name, symbol, status="error")
            return "error"

        # ── 5. No data ───────────────────────────────────────────────────────
        if new_df.empty:
            if not full_fetch:
                # Incremental fetch returned empty — could be a transient API
                # issue. Keep existing data intact and retry next run.
                logger.warning(
                    "%s: incremental fetch returned empty; keeping existing data", symbol
                )
                return "skipped"
            retry_after = date.today() + timedelta(days=self.no_data_retry_days)
            self._upsert_meta(
                provider.name, symbol,
                status="no_data",
                retry_after=retry_after.isoformat(),
            )
            return "no_data"

        # ── 6. Overlap / adj_close rewrite check ─────────────────────────────
        outcome = "updated"

        if not full_fetch:
            existing = self._load_parquet(provider.name, symbol)
            if existing is not None and not existing.empty:
                if not self._overlap_ok(existing, new_df):
                    logger.info(
                        "%s: adj_close rewrite detected in overlap; full refetch", symbol
                    )
                    try:
                        new_df = provider.fetch_daily(
                            symbol, self.config_start_date, fetch_end
                        )
                    except Exception as exc:
                        logger.error("Full refetch error for %s: %s", symbol, exc)
                        self._upsert_meta(provider.name, symbol, status="error")
                        return "error"
                    if new_df.empty:
                        retry_after = date.today() + timedelta(
                            days=self.no_data_retry_days
                        )
                        self._upsert_meta(
                            provider.name, symbol,
                            status="no_data",
                            retry_after=retry_after.isoformat(),
                        )
                        return "no_data"
                    outcome = "refetched"
                else:
                    # Merge: existing rows before overlap start + all new rows
                    cutoff = new_df.index[0]
                    existing_part = existing[existing.index < cutoff]
                    new_df = pd.concat([existing_part, new_df])
                    new_df = new_df[~new_df.index.duplicated(keep="last")]
                    new_df = new_df.sort_index()

        # ── 7. Atomic write ──────────────────────────────────────────────────
        self._write_atomic(provider.name, symbol, new_df)

        # ── 8. Update metadata ───────────────────────────────────────────────
        first = new_df.index[0]
        last = new_df.index[-1]
        self._upsert_meta(
            provider.name, symbol,
            first_date=_to_date(first).isoformat(),
            last_date=_to_date(last).isoformat(),
            rows=len(new_df),
            fetched_at=datetime.utcnow().isoformat(),
            status="ok",
            adj_fingerprint=self._fingerprint(new_df),
            retry_after=None,
        )
        return outcome

    def load(self, provider_name: str, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load from cache with consistency check.
        Returns None if not cached, status != 'ok', or file is corrupt/mismatched.
        """
        meta = self._get_meta(provider_name, symbol)
        if meta is None or meta["status"] != "ok":
            return None

        df = self._load_parquet(provider_name, symbol)
        if df is None:
            self._upsert_meta(provider_name, symbol, status="error")
            return None

        # Validate row count
        if meta["rows"] is not None and len(df) != meta["rows"]:
            logger.warning(
                "%s: row count mismatch (meta=%s, parquet=%s); invalidating",
                symbol, meta["rows"], len(df),
            )
            self._upsert_meta(provider_name, symbol, status="error")
            return None

        # Validate last date
        if meta["last_date"]:
            expected = date.fromisoformat(meta["last_date"])
            actual = _to_date(df.index[-1])
            if actual != expected:
                logger.warning(
                    "%s: last_date mismatch (meta=%s, parquet=%s); invalidating",
                    symbol, expected, actual,
                )
                self._upsert_meta(provider_name, symbol, status="error")
                return None

        return df

    def log_integrity_failure(
        self,
        run_id: str,
        symbol: str,
        check_date: Optional[date],
        check_name: str,
        detail: str,
    ) -> None:
        with self._db() as conn:
            conn.execute(
                "INSERT INTO integrity_failures "
                "(run_id, symbol, date, check_name, detail) VALUES (?,?,?,?,?)",
                (
                    run_id,
                    symbol,
                    check_date.isoformat() if check_date else None,
                    check_name,
                    detail,
                ),
            )

    def start_run(self, provider_name: str) -> str:
        run_id = str(uuid.uuid4())
        with self._db() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, started_at, provider) VALUES (?,?,?)",
                (run_id, datetime.utcnow().isoformat(), provider_name),
            )
        return run_id

    def finish_run(self, run_id: str, **counts: int) -> None:
        sets = ", ".join(f"{k}=?" for k in counts)
        sets += ", finished_at=?"
        vals = list(counts.values()) + [datetime.utcnow().isoformat(), run_id]
        with self._db() as conn:
            conn.execute(
                f"UPDATE runs SET {sets} WHERE run_id=?",
                vals,
            )

    def get_meta(self, provider_name: str, symbol: str) -> Optional[dict]:
        """Public read-only access to symbol metadata."""
        return self._get_meta(provider_name, symbol)

    def all_symbols(self, provider_name: str) -> list[str]:
        """All symbols known to the cache for this provider."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT symbol FROM symbols WHERE provider=? ORDER BY symbol",
                (provider_name,),
            ).fetchall()
        return [r[0] for r in rows]

    def latest_run_id(self) -> Optional[str]:
        """Return the run_id of the most recently finished run, or None."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT run_id FROM runs WHERE finished_at IS NOT NULL "
                "ORDER BY finished_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def all_integrity_failures(self, run_id: Optional[str] = None) -> list[dict]:
        with self._db() as conn:
            if run_id:
                rows = conn.execute(
                    "SELECT run_id, symbol, date, check_name, detail "
                    "FROM integrity_failures WHERE run_id=? ORDER BY id",
                    (run_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT run_id, symbol, date, check_name, detail "
                    "FROM integrity_failures ORDER BY id"
                ).fetchall()
        return [
            {"run_id": r[0], "symbol": r[1], "date": r[2],
             "check_name": r[3], "detail": r[4]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._db() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _get_meta(self, provider: str, symbol: str) -> Optional[dict]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM symbols WHERE symbol=? AND provider=?",
                (symbol, provider),
            ).fetchone()
        return dict(row) if row else None

    def _upsert_meta(self, provider: str, symbol: str, **fields) -> None:
        with self._db() as conn:
            existing = conn.execute(
                "SELECT symbol FROM symbols WHERE symbol=? AND provider=?",
                (symbol, provider),
            ).fetchone()
            if existing:
                if fields:
                    sets = ", ".join(f"{k}=?" for k in fields)
                    conn.execute(
                        f"UPDATE symbols SET {sets} WHERE symbol=? AND provider=?",
                        list(fields.values()) + [symbol, provider],
                    )
            else:
                cols = ["symbol", "provider"] + list(fields.keys())
                ph = ", ".join("?" * len(cols))
                conn.execute(
                    f"INSERT INTO symbols ({', '.join(cols)}) VALUES ({ph})",
                    [symbol, provider] + list(fields.values()),
                )

    def _parquet_path(self, provider: str, symbol: str) -> Path:
        return self.cache_dir / "parquet" / provider / f"{symbol}.parquet"

    def _load_parquet(self, provider: str, symbol: str) -> Optional[pd.DataFrame]:
        path = self._parquet_path(provider, symbol)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path, engine="pyarrow")
        except Exception as exc:
            logger.warning("Corrupt Parquet for %s (%s): %s", symbol, path, exc)
            return None

    def _write_atomic(self, provider: str, symbol: str, df: pd.DataFrame) -> None:
        target = self._parquet_path(provider, symbol)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, engine="pyarrow", compression="snappy")
        tmp.replace(target)

    def _overlap_ok(self, existing: pd.DataFrame, new_df: pd.DataFrame) -> bool:
        """True if adj_close values in the overlap window are within tolerance."""
        overlap_idx = existing.index.intersection(new_df.index)
        if len(overlap_idx) == 0:
            return True
        if "adj_close" not in existing.columns or "adj_close" not in new_df.columns:
            return True

        old_adj = existing.loc[overlap_idx, "adj_close"]
        new_adj = new_df.loc[overlap_idx, "adj_close"]

        common = old_adj.index.intersection(new_adj.dropna().index)
        common = common[old_adj.loc[common].notna()]
        common = common[old_adj.loc[common] != 0]

        if len(common) == 0:
            return True

        diff = ((new_adj.loc[common] - old_adj.loc[common]) / old_adj.loc[common]).abs()
        return diff.max() <= self.adj_tolerance

    @staticmethod
    def _fingerprint(df: pd.DataFrame) -> str:
        """MD5 of the last 20 adj_close values (6 decimals) for quick change detection."""
        if "adj_close" not in df.columns or df.empty:
            return ""
        # Use explicit format to avoid pandas 2.x astype(str) leaving float NaN in list
        vals = [f"{v:.6f}" if pd.notna(v) else "nan" for v in df["adj_close"].iloc[-20:]]
        return hashlib.md5(",".join(vals).encode()).hexdigest()

    def _last_completed_session(self) -> date:
        if self.calendar:
            return self.calendar.last_completed_session()
        # Fallback: yesterday
        return date.today() - timedelta(days=1)


def _to_date(ts) -> date:
    """Convert a Timestamp or date to date."""
    return ts.date() if hasattr(ts, "date") else ts
