"""Walk-forward evaluator for Engine A.

9 folds: train 5 years, test 1 year, rolling by 1 year.
  fold 0: train 2005–2009, test 2010
  fold 1: train 2006–2010, test 2011
  ...
  fold 8: train 2013–2017, test 2018

For each fold: run all 4 param combos on train, pick best Sharpe, test on test.
n_trials = 4 × 9 = 36 (used in DSR).

Param combos (Engine A):
  rsi_entry ∈ {5, 10}  ×  max_hold_days ∈ {5, 10}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

import pandas as pd

from ..data.calendar import TradingCalendar
from ..portfolio.risk import SectorMap
from ..reports.metrics import cagr, max_drawdown, sharpe
from .engine import BacktestEngine, RunResult

logger = logging.getLogger(__name__)

N_TRIALS: int = 36  # 4 combos × 9 folds

PARAM_GRID = [
    {"rsi_entry": 5, "max_hold_days": 5},
    {"rsi_entry": 5, "max_hold_days": 10},
    {"rsi_entry": 10, "max_hold_days": 5},
    {"rsi_entry": 10, "max_hold_days": 10},
]


@dataclass
class WalkForwardFold:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    best_params: dict
    train_sharpe: float
    test_sharpe: float
    test_cagr: float
    test_max_dd: float
    train_result: Optional[RunResult] = None
    test_result: Optional[RunResult] = None


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    n_trials: int = N_TRIALS

    @property
    def avg_test_sharpe(self) -> float:
        valid = [f.test_sharpe for f in self.folds if not pd.isna(f.test_sharpe)]
        return float(pd.Series(valid).mean()) if valid else float("nan")

    @property
    def avg_test_cagr(self) -> float:
        valid = [f.test_cagr for f in self.folds if not pd.isna(f.test_cagr)]
        return float(pd.Series(valid).mean()) if valid else float("nan")

    @property
    def best_params_frequency(self) -> dict:
        freq: dict[str, int] = {}
        for f in self.folds:
            key = str(f.best_params)
            freq[key] = freq.get(key, 0) + 1
        return freq


def _make_folds(dev_start: date = date(2005, 1, 1), dev_end: date = date(2018, 12, 31)) -> list[tuple[date, date, date, date]]:
    """Generate 9 (train_start, train_end, test_start, test_end) tuples."""
    folds = []
    for i in range(9):
        train_start = date(2005 + i, 1, 1)
        train_end = date(2009 + i, 12, 31)
        test_start = date(2010 + i, 1, 1)
        test_end = date(2010 + i, 12, 31)
        folds.append((train_start, train_end, test_start, test_end))
    return folds


class WalkForwardEvaluator:
    def __init__(
        self,
        price_data: dict,
        calendar: TradingCalendar,
        universe_members: Callable,
        initial_capital: float,
        base_engine_cfg,
        risk_cfg,
        cost_cfg,
        sector_map: Optional[SectorMap] = None,
    ) -> None:
        self._data = price_data
        self._calendar = calendar
        self._members_fn = universe_members
        self._initial_capital = initial_capital
        self._base_eng = base_engine_cfg
        self._risk = risk_cfg
        self._cost = cost_cfg
        self._sector_map = sector_map

    def _run_with_params(self, params: dict, start: date, end: date) -> RunResult:
        from ..config import EngineAConfig
        eng_cfg = EngineAConfig(
            rsi_entry=params["rsi_entry"],
            max_hold_days=params["max_hold_days"],
            spy_symbol=self._base_eng.spy_symbol,
        )
        engine = BacktestEngine(
            price_data=self._data,
            calendar=self._calendar,
            universe_members=self._members_fn,
            initial_capital=self._initial_capital,
            engine_cfg=eng_cfg,
            risk_cfg=self._risk,
            cost_cfg=self._cost,
            sector_map=self._sector_map,
        )
        return engine.run(start, end)

    def run(self) -> WalkForwardResult:
        folds_spec = _make_folds()
        result_folds: list[WalkForwardFold] = []

        for i, (ts, te, vs, ve) in enumerate(folds_spec):
            logger.info("Fold %d/%d: train %s–%s, test %s–%s",
                        i + 1, len(folds_spec), ts, te, vs, ve)

            # Train: evaluate all param combos
            best_params = PARAM_GRID[0]
            best_train_sharpe = float("-inf")
            best_train_result = None

            for params in PARAM_GRID:
                try:
                    tr_result = self._run_with_params(params, ts, te)
                    tr_sharpe = sharpe(tr_result.equity_curve["equity"])
                    if pd.isna(tr_sharpe):
                        tr_sharpe = float("-inf")
                    if tr_sharpe > best_train_sharpe:
                        best_train_sharpe = tr_sharpe
                        best_params = params
                        best_train_result = tr_result
                except Exception as exc:
                    logger.warning("Train fold %d params %s failed: %s", i, params, exc)

            # Test: run best params on test period
            try:
                test_result = self._run_with_params(best_params, vs, ve)
                test_eq = test_result.equity_curve["equity"]
                test_sharpe_val = sharpe(test_eq)
                test_cagr_val = cagr(test_eq)
                test_mdd_val = max_drawdown(test_eq)
            except Exception as exc:
                logger.warning("Test fold %d failed: %s", i, exc)
                test_result = None
                test_sharpe_val = float("nan")
                test_cagr_val = float("nan")
                test_mdd_val = float("nan")

            fold = WalkForwardFold(
                fold_index=i,
                train_start=ts,
                train_end=te,
                test_start=vs,
                test_end=ve,
                best_params=best_params,
                train_sharpe=best_train_sharpe if best_train_sharpe > float("-inf") else float("nan"),
                test_sharpe=test_sharpe_val,
                test_cagr=test_cagr_val,
                test_max_dd=test_mdd_val,
                train_result=best_train_result,
                test_result=test_result,
            )
            result_folds.append(fold)
            logger.info(
                "  Best params: %s | Train Sharpe: %.3f | Test Sharpe: %.3f",
                best_params, fold.train_sharpe, fold.test_sharpe,
            )

        return WalkForwardResult(folds=result_folds, n_trials=N_TRIALS)
