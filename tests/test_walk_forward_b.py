"""Tests for WalkForwardEvaluatorB and WalkForwardCombined.

Verifies: fold count, dates, non-overlap, n_trials=27 for B, n_trials=9 for combined.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.walk_forward import _make_folds, WalkForwardResult, WalkForwardFold
from src.backtest.walk_forward_b import N_TRIALS_B, PARAM_GRID_B, WalkForwardEvaluatorB
from src.backtest.walk_forward_combined import N_TRIALS_COMBINED, WalkForwardCombined


# ── fold structure tests ──────────────────────────────────────────────────────


class TestFoldStructure:
    def test_nine_folds(self):
        folds = _make_folds()
        assert len(folds) == 9

    def test_train_five_years(self):
        folds = _make_folds()
        for ts, te, vs, ve in folds:
            assert (te.year - ts.year) == 4  # 5-year span: e.g. 2005–2009

    def test_test_one_year(self):
        folds = _make_folds()
        for ts, te, vs, ve in folds:
            assert vs.year == ve.year  # same calendar year

    def test_no_overlap_between_train_and_test(self):
        folds = _make_folds()
        for ts, te, vs, ve in folds:
            assert te < vs  # train ends before test starts

    def test_no_overlap_between_consecutive_test_periods(self):
        folds = _make_folds()
        for i in range(len(folds) - 1):
            _, _, _, ve_i = folds[i]
            _, _, vs_j, _ = folds[i + 1]
            assert ve_i < vs_j

    def test_folds_cover_development_period(self):
        folds = _make_folds()
        first_train_start = folds[0][0]
        last_test_end = folds[-1][3]
        assert first_train_start == date(2005, 1, 1)
        assert last_test_end == date(2018, 12, 31)


# ── param grid ───────────────────────────────────────────────────────────────


class TestParamGridB:
    def test_n_trials_is_27(self):
        assert N_TRIALS_B == 27

    def test_three_param_combos(self):
        assert len(PARAM_GRID_B) == 3

    def test_top_n_values(self):
        top_n_vals = {p["top_n"] for p in PARAM_GRID_B}
        assert top_n_vals == {5, 10, 15}

    def test_n_trials_equals_combos_times_folds(self):
        assert N_TRIALS_B == len(PARAM_GRID_B) * len(_make_folds())


# ── n_trials combined ────────────────────────────────────────────────────────


class TestNTrialsCombined:
    def test_n_trials_is_9(self):
        assert N_TRIALS_COMBINED == 9

    def test_n_trials_equals_n_folds(self):
        assert N_TRIALS_COMBINED == len(_make_folds())


# ── WalkForwardEvaluatorB with mock data ─────────────────────────────────────


def _build_minimal_price_data(n_tickers=15, n_sessions=380, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    price_data = {}
    idx = pd.date_range("2005-01-01", periods=n_sessions, freq="B")

    for i in range(n_tickers):
        close = 100 + rng.random(n_sessions).cumsum() * 0.1
        close = close.clip(min=1)
        price_data[f"T{i:02d}"] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj_close": close,
                "volume": [5_000_000.0] * n_sessions,
                "dividends": [0.0] * n_sessions,
                "splits": [0.0] * n_sessions,
            },
            index=idx,
        )

    spy_close = 100 + rng.random(n_sessions).cumsum() * 0.2
    spy_close = spy_close.clip(min=1)
    price_data["SPY"] = pd.DataFrame(
        {
            "open": spy_close,
            "high": spy_close * 1.01,
            "low": spy_close * 0.99,
            "close": spy_close,
            "adj_close": spy_close,
            "volume": [1e8] * n_sessions,
            "dividends": [0.0] * n_sessions,
            "splits": [0.0] * n_sessions,
        },
        index=idx,
    )

    from src.data.calendar import TradingCalendar
    calendar = TradingCalendar.from_dates([d.date() for d in idx])
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    def members_fn(d):
        return set(tickers)

    return price_data, calendar, members_fn


class TestWalkForwardEvaluatorBStructure:
    """Tests for the WalkForwardEvaluatorB that verify structure without a full run."""

    def test_n_trials_attribute(self):
        assert N_TRIALS_B == 27

    def test_param_grid_contains_all_top_n(self):
        top_ns = [p["top_n"] for p in PARAM_GRID_B]
        assert sorted(top_ns) == [5, 10, 15]


class TestWalkForwardCombinedStructure:
    """Tests for WalkForwardCombined fold wiring without running a full backtest."""

    def _make_mock_wf_result(self, n_folds=9, default_params_a=None, default_params_b=None):
        if default_params_a is None:
            default_params_a = {"rsi_entry": 10, "max_hold_days": 10}
        if default_params_b is None:
            default_params_b = {"top_n": 10}

        folds_spec = _make_folds()
        folds = [
            WalkForwardFold(
                fold_index=i,
                train_start=ts,
                train_end=te,
                test_start=vs,
                test_end=ve,
                best_params=dict(default_params_a),
                train_sharpe=0.5,
                test_sharpe=0.4,
                test_cagr=0.05,
                test_max_dd=0.10,
            )
            for i, (ts, te, vs, ve) in enumerate(folds_spec)
        ]
        return WalkForwardResult(folds=folds, n_trials=36)

    def _make_mock_wf_b_result(self):
        folds_spec = _make_folds()
        folds = [
            WalkForwardFold(
                fold_index=i,
                train_start=ts,
                train_end=te,
                test_start=vs,
                test_end=ve,
                best_params={"top_n": 10},
                train_sharpe=0.4,
                test_sharpe=0.3,
                test_cagr=0.04,
                test_max_dd=0.12,
            )
            for i, (ts, te, vs, ve) in enumerate(folds_spec)
        ]
        return WalkForwardResult(folds=folds, n_trials=27)

    def test_mismatched_folds_raises(self):
        from src.config import EngineAConfig, EngineBConfig, RiskConfig, CostConfig
        from src.data.calendar import TradingCalendar

        calendar = TradingCalendar.from_dates([date(2010, 1, 4)])

        wf_a = self._make_mock_wf_result()
        # Wrong number of folds for B
        bad_wf_b = WalkForwardResult(folds=wf_a.folds[:5], n_trials=15)

        evaluator = WalkForwardCombined(
            price_data={},
            calendar=calendar,
            universe_members=lambda d: set(),
            initial_capital=100_000.0,
            base_engine_a_cfg=EngineAConfig(),
            base_engine_b_cfg=EngineBConfig(),
            risk_cfg=RiskConfig(),
            cost_cfg=CostConfig(),
            wf_a_result=wf_a,
            wf_b_result=bad_wf_b,
        )
        with pytest.raises(ValueError, match="folds"):
            evaluator.run()

    def test_n_trials_combined_is_9(self):
        assert N_TRIALS_COMBINED == 9
