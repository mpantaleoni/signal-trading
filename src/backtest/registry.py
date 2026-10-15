"""Experiment registry — SQLite-backed log of every backtest run.

Schema
------
experiments: one row per backtest run
  id            INTEGER PRIMARY KEY
  run_id        TEXT NOT NULL (UUID)
  engine        TEXT
  start_date    TEXT
  end_date      TEXT
  params_json   TEXT (JSON serialisation of engine + risk params)
  git_commit    TEXT
  run_at        TEXT (ISO-8601 timestamp)
  cagr          REAL
  sharpe        REAL
  max_drawdown  REAL
  total_return  REAL
  n_trades      INTEGER

walk_forward_folds: one row per fold × run
  id            INTEGER PRIMARY KEY
  experiment_id INTEGER REFERENCES experiments(id)
  fold_index    INTEGER
  train_start   TEXT
  train_end     TEXT
  test_start    TEXT
  test_end      TEXT
  train_sharpe  REAL
  test_sharpe   REAL
  test_cagr     REAL
  test_max_dd   REAL
  best_params   TEXT (JSON)
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional


class ExperimentRegistry:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id        TEXT NOT NULL,
                engine        TEXT,
                start_date    TEXT,
                end_date      TEXT,
                params_json   TEXT,
                git_commit    TEXT,
                run_at        TEXT,
                cagr          REAL,
                sharpe        REAL,
                max_drawdown  REAL,
                total_return  REAL,
                n_trades      INTEGER
            );
            CREATE TABLE IF NOT EXISTS walk_forward_folds (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER REFERENCES experiments(id),
                fold_index    INTEGER,
                train_start   TEXT,
                train_end     TEXT,
                test_start    TEXT,
                test_end      TEXT,
                train_sharpe  REAL,
                test_sharpe   REAL,
                test_cagr     REAL,
                test_max_dd   REAL,
                best_params   TEXT
            );
            """
        )
        self._conn.commit()

    def log_run(
        self,
        engine: str,
        start_date: date,
        end_date: date,
        params: dict[str, Any],
        metrics: dict[str, Any],
        git_commit: str = "",
    ) -> int:
        """Insert one experiment record. Returns the row id."""
        run_id = str(uuid.uuid4())
        cur = self._conn.execute(
            """
            INSERT INTO experiments
              (run_id, engine, start_date, end_date, params_json, git_commit,
               run_at, cagr, sharpe, max_drawdown, total_return, n_trades)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                engine,
                start_date.isoformat(),
                end_date.isoformat(),
                json.dumps(params),
                git_commit,
                datetime.utcnow().isoformat(),
                metrics.get("cagr"),
                metrics.get("sharpe"),
                metrics.get("max_drawdown"),
                metrics.get("total_return"),
                metrics.get("n_trades"),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def log_fold(
        self,
        experiment_id: int,
        fold_index: int,
        train_start: date,
        train_end: date,
        test_start: date,
        test_end: date,
        train_sharpe: float,
        test_sharpe: float,
        test_cagr: float,
        test_max_dd: float,
        best_params: dict[str, Any],
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO walk_forward_folds
              (experiment_id, fold_index, train_start, train_end,
               test_start, test_end, train_sharpe, test_sharpe,
               test_cagr, test_max_dd, best_params)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                experiment_id,
                fold_index,
                train_start.isoformat(),
                train_end.isoformat(),
                test_start.isoformat(),
                test_end.isoformat(),
                train_sharpe,
                test_sharpe,
                test_cagr,
                test_max_dd,
                json.dumps(best_params),
            ),
        )
        self._conn.commit()

    def count_experiments(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    def all_sharpes(self) -> list[float]:
        rows = self._conn.execute(
            "SELECT sharpe FROM experiments WHERE sharpe IS NOT NULL"
        ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()
