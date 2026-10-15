"""Tests for CAGR, Sharpe, Sortino, drawdown, DSR with synthetic equity curves."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.reports.metrics import (
    _acklam_inv_normal,
    _normal_cdf,
    avg_holding_days,
    avg_loss,
    avg_win,
    cagr,
    deflated_sharpe_ratio,
    max_drawdown,
    max_drawdown_duration,
    sharpe,
    sortino,
    total_return,
    win_rate,
    yearly_returns,
)


def _equity(values, start="2010-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


class TestCagr:
    def test_flat_equity(self):
        eq = _equity([100.0] * 253)
        assert cagr(eq) == pytest.approx(0.0, abs=1e-6)

    def test_doubles_in_one_year(self):
        eq = _equity([100.0, 200.0], start="2010-01-01")
        result = cagr(eq)
        # 2 data points separated by 1 business day → not 1 year
        # Just check it's positive and large
        assert result > 0

    def test_known_cagr(self):
        # Start 2010-01-04, end 2020-01-03 ≈ 10 years, 2× growth → CAGR ≈ 7.18%
        start = "2010-01-04"
        end = "2020-01-03"
        idx = pd.date_range(start, end, freq="B")
        n = len(idx)
        vals = np.linspace(100.0, 200.0, n)
        eq = pd.Series(vals, index=idx)
        result = cagr(eq)
        assert result == pytest.approx((200 / 100) ** (1 / 10) - 1, rel=0.05)

    def test_short_series_returns_nan(self):
        eq = _equity([100.0])
        assert math.isnan(cagr(eq))

    def test_zero_start_returns_nan(self):
        eq = _equity([0.0, 100.0])
        assert math.isnan(cagr(eq))


class TestSharpe:
    def test_all_same_returns_nan(self):
        eq = _equity([100.0] * 50)
        result = sharpe(eq)
        assert math.isnan(result) or math.isinf(result)

    def test_positive_returns_positive_sharpe(self):
        vals = [100.0 * (1.001 ** i) for i in range(252)]
        eq = _equity(vals)
        result = sharpe(eq)
        assert result > 0

    def test_negative_drift_negative_sharpe(self):
        vals = [100.0 * (0.999 ** i) for i in range(252)]
        eq = _equity(vals)
        result = sharpe(eq)
        assert result < 0


class TestSortino:
    def test_no_losses_gives_inf(self):
        vals = [100.0 * (1.001 ** i) for i in range(252)]
        eq = _equity(vals)
        result = sortino(eq)
        assert math.isinf(result) or result > 10

    def test_higher_than_sharpe_for_positive_skew(self):
        # Only upside volatility → Sortino > Sharpe
        vals = [100.0 + i * 0.1 + (0.5 if i % 5 == 0 else 0) for i in range(252)]
        eq = _equity(vals)
        s = sharpe(eq)
        so = sortino(eq)
        if not math.isnan(s) and not math.isnan(so) and not math.isinf(so):
            assert so >= s


class TestMaxDrawdown:
    def test_no_drawdown(self):
        vals = [100.0 * (1.001 ** i) for i in range(100)]
        eq = _equity(vals)
        assert max_drawdown(eq) == pytest.approx(0.0, abs=1e-6)

    def test_known_drawdown(self):
        # Peak 200, trough 100 → 50% drawdown
        vals = [100, 150, 200, 150, 100, 120, 180]
        eq = _equity(vals)
        assert max_drawdown(eq) == pytest.approx(0.50, rel=0.01)

    def test_short_series(self):
        eq = _equity([100.0])
        assert max_drawdown(eq) == 0.0

    def test_drawdown_within_0_to_1(self):
        rng = np.random.default_rng(42)
        vals = 100 + np.cumsum(rng.normal(0, 1, 200))
        vals = np.maximum(vals, 1.0)
        eq = _equity(vals)
        dd = max_drawdown(eq)
        assert 0.0 <= dd <= 1.0


class TestMaxDrawdownDuration:
    def test_no_drawdown_gives_zero(self):
        vals = [100.0 * (1.001 ** i) for i in range(100)]
        eq = _equity(vals)
        assert max_drawdown_duration(eq) == 0

    def test_duration_is_positive(self):
        vals = [100, 150, 200, 150, 100, 120, 180, 210]
        eq = _equity(vals)
        assert max_drawdown_duration(eq) > 0


class TestTotalReturn:
    def test_doubled_equity(self):
        eq = _equity([100.0, 200.0])
        assert total_return(eq) == pytest.approx(1.0)

    def test_halved_equity(self):
        eq = _equity([100.0, 50.0])
        assert total_return(eq) == pytest.approx(-0.50)

    def test_flat_equity(self):
        eq = _equity([100.0] * 10)
        assert total_return(eq) == pytest.approx(0.0)


class TestYearlyReturns:
    def test_two_year_series(self):
        idx = pd.date_range("2010-01-04", "2011-12-30", freq="B")
        n = len(idx)
        vals = np.linspace(100.0, 120.0, n)
        eq = pd.Series(vals, index=idx)
        yr = yearly_returns(eq)
        assert 2010 in yr
        assert 2011 in yr

    def test_returns_are_finite(self):
        idx = pd.date_range("2010-01-04", "2012-12-31", freq="B")
        n = len(idx)
        vals = np.linspace(100.0, 150.0, n)
        eq = pd.Series(vals, index=idx)
        yr = yearly_returns(eq)
        for v in yr.values():
            assert math.isfinite(v)


class TestDeflatedSharpe:
    def test_high_sharpe_few_trials(self):
        # High observed Sharpe, few trials → DSR near 1
        dsr = deflated_sharpe_ratio(3.0, n_trials=4, n_observations=252)
        assert not math.isnan(dsr)
        assert dsr > 0.5

    def test_negative_sharpe_gives_low_dsr(self):
        dsr = deflated_sharpe_ratio(-2.0, n_trials=36, n_observations=252)
        assert not math.isnan(dsr)
        assert dsr < 0.1

    def test_insufficient_data(self):
        dsr = deflated_sharpe_ratio(1.0, n_trials=36, n_observations=2)
        assert math.isnan(dsr)

    def test_result_between_0_and_1(self):
        dsr = deflated_sharpe_ratio(1.0, n_trials=36, n_observations=252)
        assert 0.0 <= dsr <= 1.0

    def test_acklam_known_values(self):
        assert _acklam_inv_normal(0.5) == pytest.approx(0.0, abs=1e-6)
        assert _acklam_inv_normal(0.975) == pytest.approx(1.96, abs=0.01)

    def test_normal_cdf_known_values(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5, abs=1e-6)
        assert _normal_cdf(1.96) == pytest.approx(0.975, abs=0.01)


class TestTradeLevelMetrics:
    def _trades(self):
        return [
            {"pnl_usd": 100.0, "sessions_held": 5},
            {"pnl_usd": -50.0, "sessions_held": 3},
            {"pnl_usd": 200.0, "sessions_held": 7},
            {"pnl_usd": -30.0, "sessions_held": 2},
        ]

    def test_win_rate(self):
        wr = win_rate(self._trades())
        assert wr == pytest.approx(0.5)

    def test_avg_win(self):
        aw = avg_win(self._trades())
        assert aw == pytest.approx(150.0)

    def test_avg_loss(self):
        al = avg_loss(self._trades())
        assert al == pytest.approx(-40.0)

    def test_avg_holding_days(self):
        ahd = avg_holding_days(self._trades())
        assert ahd == pytest.approx((5 + 3 + 7 + 2) / 4)

    def test_empty_trades(self):
        assert math.isnan(win_rate([]))
        assert math.isnan(avg_holding_days([]))
        assert avg_win([]) == 0.0
        assert avg_loss([]) == 0.0
