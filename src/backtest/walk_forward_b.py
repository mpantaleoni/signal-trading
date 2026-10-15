"""Walk-forward evaluator for Engine B.

Reuses the same 9 folds as Engine A.
Param grid: top_n ∈ {5, 10, 15} → 3 combos × 9 folds = 27 n_trials.
Selects best top_n by in-sample Sharpe per fold; tests on held-out year.
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
from .engine import RunResult
from .engine_b import EngineBBacktest
from .walk_forward import WalkForwardFold, WalkForwardResult, _make_folds

logger = logging.getLogger(__name__)

N_TRIALS_B: int = 27  # 3 combos × 9 folds

PARAM_GRID_B = [
    {"top_n": 5},
    {"top_n": 10},
    {"top_n": 15},
]


class WalkForwardEvaluatorB:
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
        from ..config import EngineBConfig
        eng_cfg = EngineBConfig(
            top_n=params["top_n"],
            activity_lookback_fast=self._base_eng.activity_lookback_fast,
            activity_lookback_slow=self._base_eng.activity_lookback_slow,
            momentum_lookback=self._base_eng.momentum_lookback,
            momentum_skip=self._base_eng.momentum_skip,
            spy_symbol=self._base_eng.spy_symbol,
        )
        engine = EngineBBacktest(
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
            logger.info("Engine B Fold %d/%d: train %s–%s, test %s–%s",
                        i + 1, len(folds_spec), ts, te, vs, ve)

            best_params = PARAM_GRID_B[0]
            best_train_sharpe = float("-inf")
            best_train_result = None

            for params in PARAM_GRID_B:
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
                    logger.warning("Engine B train fold %d params %s failed: %s", i, params, exc)

            try:
                test_result = self._run_with_params(best_params, vs, ve)
                test_eq = test_result.equity_curve["equity"]
                test_sharpe_val = sharpe(test_eq)
                test_cagr_val = cagr(test_eq)
                test_mdd_val = max_drawdown(test_eq)
            except Exception as exc:
                logger.warning("Engine B test fold %d failed: %s", i, exc)
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

        return WalkForwardResult(folds=result_folds, n_trials=N_TRIALS_B)
