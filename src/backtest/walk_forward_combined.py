"""Walk-forward evaluator for the Combined Engine (A + B).

Takes two WalkForwardResult objects (from WF_A and WF_B).
For fold i: uses wf_a.folds[i].best_params + wf_b.folds[i].best_params.
Runs CombinedBacktest on the test period.
n_trials = 9 (no additional search; params already selected individually).
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
from .combined_engine import CombinedBacktest
from .engine import RunResult
from .walk_forward import WalkForwardFold, WalkForwardResult, _make_folds

logger = logging.getLogger(__name__)

N_TRIALS_COMBINED: int = 9  # no grid search; params come from WF_A and WF_B


class WalkForwardCombined:
    """Evaluate the combined engine using the best params from WF_A and WF_B per fold.

    Parameters
    ----------
    price_data      : dict[ticker, DataFrame]
    calendar        : TradingCalendar
    universe_members: callable(date) -> set[str]
    initial_capital : float
    base_engine_a_cfg : EngineAConfig  (only spy_symbol used as fallback)
    base_engine_b_cfg : EngineBConfig  (activity/momentum params, spy_symbol)
    risk_cfg        : RiskConfig
    cost_cfg        : CostConfig
    wf_a_result     : WalkForwardResult from Engine A walk-forward
    wf_b_result     : WalkForwardResult from Engine B walk-forward
    sector_map      : optional SectorMap
    """

    def __init__(
        self,
        price_data: dict,
        calendar: TradingCalendar,
        universe_members: Callable,
        initial_capital: float,
        base_engine_a_cfg,
        base_engine_b_cfg,
        risk_cfg,
        cost_cfg,
        wf_a_result: WalkForwardResult,
        wf_b_result: WalkForwardResult,
        sector_map: Optional[SectorMap] = None,
    ) -> None:
        self._data = price_data
        self._calendar = calendar
        self._members_fn = universe_members
        self._initial_capital = initial_capital
        self._base_a = base_engine_a_cfg
        self._base_b = base_engine_b_cfg
        self._risk = risk_cfg
        self._cost = cost_cfg
        self._wf_a = wf_a_result
        self._wf_b = wf_b_result
        self._sector_map = sector_map

    def _run_fold(self, params_a: dict, params_b: dict, start: date, end: date) -> RunResult:
        from ..config import EngineAConfig, EngineBConfig
        eng_a_cfg = EngineAConfig(
            rsi_entry=params_a["rsi_entry"],
            max_hold_days=params_a["max_hold_days"],
            spy_symbol=self._base_a.spy_symbol,
        )
        eng_b_cfg = EngineBConfig(
            top_n=params_b["top_n"],
            activity_lookback_fast=self._base_b.activity_lookback_fast,
            activity_lookback_slow=self._base_b.activity_lookback_slow,
            momentum_lookback=self._base_b.momentum_lookback,
            momentum_skip=self._base_b.momentum_skip,
            spy_symbol=self._base_b.spy_symbol,
        )
        engine = CombinedBacktest(
            price_data=self._data,
            calendar=self._calendar,
            universe_members=self._members_fn,
            initial_capital=self._initial_capital,
            engine_a_cfg=eng_a_cfg,
            engine_b_cfg=eng_b_cfg,
            risk_cfg=self._risk,
            cost_cfg=self._cost,
            sector_map=self._sector_map,
        )
        return engine.run(start, end)

    def run(self) -> WalkForwardResult:
        folds_spec = _make_folds()
        if len(self._wf_a.folds) != len(folds_spec) or len(self._wf_b.folds) != len(folds_spec):
            raise ValueError(
                f"WF_A has {len(self._wf_a.folds)} folds, WF_B has {len(self._wf_b.folds)}, "
                f"expected {len(folds_spec)}"
            )

        result_folds: list[WalkForwardFold] = []

        for i, (ts, te, vs, ve) in enumerate(folds_spec):
            params_a = self._wf_a.folds[i].best_params
            params_b = self._wf_b.folds[i].best_params

            logger.info(
                "Combined fold %d/%d: test %s–%s | A=%s B=%s",
                i + 1, len(folds_spec), vs, ve, params_a, params_b,
            )

            try:
                test_result = self._run_fold(params_a, params_b, vs, ve)
                test_eq = test_result.equity_curve["equity"]
                test_sharpe_val = sharpe(test_eq)
                test_cagr_val = cagr(test_eq)
                test_mdd_val = max_drawdown(test_eq)
            except Exception as exc:
                logger.warning("Combined fold %d failed: %s", i, exc)
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
                best_params={"a": params_a, "b": params_b},
                train_sharpe=float("nan"),  # no train run for combined
                test_sharpe=test_sharpe_val,
                test_cagr=test_cagr_val,
                test_max_dd=test_mdd_val,
                train_result=None,
                test_result=test_result,
            )
            result_folds.append(fold)
            logger.info(
                "  Test Sharpe: %.3f | Test CAGR: %.1f%% | Max DD: %.1f%%",
                fold.test_sharpe,
                fold.test_cagr * 100 if not pd.isna(fold.test_cagr) else float("nan"),
                fold.test_max_dd * 100 if not pd.isna(fold.test_max_dd) else float("nan"),
            )

        return WalkForwardResult(folds=result_folds, n_trials=N_TRIALS_COMBINED)
