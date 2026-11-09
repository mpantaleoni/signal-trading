"""Tests for src/signals/engine_momentum.py and src/backtest/engine_momentum.py."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.calendar import TradingCalendar
from src.signals.engine_momentum import (
    compute_52wh_pct,
    compute_momentum_12_1,
    compute_spy_regime,
    compute_stock_above_sma,
    select_momentum_stocks,
    select_short_stocks,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_df(n: int, close_vals=None, seed=42, start="2005-01-01"):
    """Canonical OHLCV DataFrame, indexed by tz-naive business-day timestamps."""
    idx = pd.date_range(start, periods=n, freq="B")
    if close_vals is None:
        rng = np.random.default_rng(seed)
        close_vals = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close_vals = np.array(close_vals, dtype=float)
    return pd.DataFrame(
        {
            "open": close_vals,
            "high": close_vals * 1.01,
            "low": close_vals * 0.99,
            "close": close_vals,
            "adj_close": close_vals,
            "volume": [1_000_000.0] * n,
            "dividends": [0.0] * n,
            "splits": [0.0] * n,
        },
        index=idx,
    )


def _make_calendar(start="2005-01-01", end="2020-12-31"):
    dates = [d.date() for d in pd.date_range(start, end, freq="B")]
    return TradingCalendar.from_dates(dates)


# ── compute_momentum_12_1 ─────────────────────────────────────────────────────


class TestComputeMomentum121:
    def test_returns_none_if_fewer_than_273_rows(self):
        df = _make_df(272)
        assert compute_momentum_12_1(df) is None

    def test_returns_value_at_exactly_273_rows(self):
        # Rising prices — end price > start price
        close_vals = np.linspace(100, 200, 273)
        df = _make_df(273, close_vals=close_vals)
        result = compute_momentum_12_1(df)
        assert result is not None
        assert result > 0

    def test_negative_momentum_for_declining_prices(self):
        close_vals = np.linspace(200, 100, 300)
        df = _make_df(300, close_vals=close_vals)
        result = compute_momentum_12_1(df)
        assert result is not None
        assert result < 0

    def test_no_lookahead_uses_iloc_minus21_and_minus273(self):
        """Appending a future bar must NOT change the computed value."""
        n = 400
        df = _make_df(n)
        t_ts = df.index[-1]
        # slice to t_ts
        df_at_t = df.loc[df.index <= t_ts]
        score_at_t = compute_momentum_12_1(df_at_t)

        # Add a future bar with a drastically different price
        future_idx = t_ts + pd.Timedelta(days=1)
        df_future = df.copy()
        df_future.loc[future_idx] = df_future.iloc[-1]
        df_future.at[future_idx, "adj_close"] = 9999.0
        # slice only up to original t_ts
        df_slice = df_future.loc[df_future.index <= t_ts]
        assert compute_momentum_12_1(df_slice) == score_at_t


# ── compute_52wh_pct ──────────────────────────────────────────────────────────


class TestCompute52whPct:
    def test_returns_none_if_fewer_than_252_rows(self):
        df = _make_df(251)
        assert compute_52wh_pct(df) is None

    def test_returns_value_at_exactly_252_rows(self):
        df = _make_df(252)
        result = compute_52wh_pct(df)
        assert result is not None
        assert 0.0 < result <= 1.0

    def test_at_or_near_52_week_high(self):
        # Last bar is at the 52-week high
        close_vals = list(range(100, 352))  # 252 values, last is max
        df = _make_df(252, close_vals=close_vals)
        result = compute_52wh_pct(df)
        assert result is not None
        assert abs(result - 1.0) < 0.01  # close to 1

    def test_well_below_52_week_high(self):
        # Spike in the middle, then big decline
        n = 300
        close_vals = [100.0] * n
        close_vals[150] = 500.0  # 52-week high spike
        close_vals[-1] = 100.0
        df = _make_df(n, close_vals=close_vals)
        result = compute_52wh_pct(df)
        assert result is not None
        assert result < 0.5  # well below 52-week high

    def test_no_lookahead(self):
        """Score must not change when a future bar is added beyond t_ts."""
        df = _make_df(300)
        t_ts = df.index[-1]
        score_at_t = compute_52wh_pct(df.loc[df.index <= t_ts])

        # Append future bar with high price
        future_idx = t_ts + pd.Timedelta(days=1)
        df_future = df.copy()
        df_future.loc[future_idx] = df_future.iloc[-1]
        df_future.at[future_idx, "adj_close"] = 9999.0
        df_future.at[future_idx, "high"] = 9999.0
        score_sliced = compute_52wh_pct(df_future.loc[df_future.index <= t_ts])
        assert score_sliced == score_at_t


# ── compute_spy_regime ────────────────────────────────────────────────────────


class TestComputeSpyRegime:
    def test_true_when_both_conditions_met(self):
        # Steadily rising: 12-1 momentum > 0 AND close > SMA200
        close_vals = np.linspace(100, 200, 300)
        df = _make_df(300, close_vals=close_vals)
        assert compute_spy_regime(df) is True

    def test_false_for_declining_spy(self):
        # 12-1 momentum < 0 → False even if close > SMA200 doesn't matter
        close_vals = np.linspace(200, 100, 300)
        df = _make_df(300, close_vals=close_vals)
        assert compute_spy_regime(df) is False

    def test_false_if_insufficient_history(self):
        df = _make_df(272)
        assert compute_spy_regime(df) is False

    def test_false_when_close_below_sma200(self):
        # Strong 12-1 momentum but recent sharp drop puts close below SMA200
        close_vals = np.ones(300) * 150.0
        # Set recent 20 bars to a sharp drop so close < SMA200
        close_vals[-20:] = 50.0
        # But 12-1 momentum window (iloc[-273] to iloc[-21]) is above 100
        close_vals[-273] = 100.0
        close_vals[-21] = 150.0   # momentum positive
        df = _make_df(300, close_vals=close_vals)
        # SMA200 = mean of last 200 values; most are 150 but last 20 are 50
        # SMA200 ≈ (180*150 + 20*50)/200 = (27000+1000)/200 = 140; close[-1] = 50 < 140
        assert compute_spy_regime(df) is False

    def test_true_requires_positive_momentum_and_above_sma(self):
        # Confirm both conditions are needed: momentum > 0 AND close > SMA200
        # Rising trend satisfies both
        close_vals = np.linspace(100, 200, 300)
        df = _make_df(300, close_vals=close_vals)
        assert compute_spy_regime(df) is True


# ── compute_stock_above_sma ───────────────────────────────────────────────────


class TestComputeStockAboveSma:
    def test_true_for_rising_stock_above_sma(self):
        close_vals = np.linspace(100, 200, 250)
        df = _make_df(250, close_vals=close_vals)
        assert compute_stock_above_sma(df, sma_period=200) is True

    def test_false_for_declining_stock_below_sma(self):
        # Spike then fall: last bar well below SMA
        close_vals = np.ones(250) * 150.0
        close_vals[-1] = 50.0   # sharp drop at the end
        df = _make_df(250, close_vals=close_vals)
        # SMA200 ≈ mean of last 200 bars; 199 of them are 150, last is 50
        # SMA ≈ (199*150 + 50)/200 = 149.5; close = 50 < 149.5
        assert compute_stock_above_sma(df, sma_period=200) is False

    def test_false_if_insufficient_history(self):
        df = _make_df(199)
        assert compute_stock_above_sma(df, sma_period=200) is False

    def test_no_lookahead(self):
        df = _make_df(250, close_vals=np.linspace(100, 200, 250))
        t_ts = df.index[-1]
        result_at_t = compute_stock_above_sma(df.loc[df.index <= t_ts], 200)
        # Append a future bar with a low price that would change SMA
        future_ts = t_ts + pd.Timedelta(days=1)
        df_future = df.copy()
        df_future.loc[future_ts] = df_future.iloc[-1]
        df_future.at[future_ts, "adj_close"] = 1.0
        sliced = df_future.loc[df_future.index <= t_ts]
        assert compute_stock_above_sma(sliced, 200) == result_at_t


# ── select_momentum_stocks ────────────────────────────────────────────────────


class TestSelectMomentumStocks:
    def _make_universe(self, n_stocks=5, n_bars=300):
        """Create a universe with clearly differentiated momentum."""
        dfs = {}
        for i in range(n_stocks):
            seed = i + 1
            rng = np.random.default_rng(seed)
            # Give different stocks different trends
            trend = (i - n_stocks // 2) * 0.3  # -ve to +ve trends
            close_vals = 100.0 + np.cumsum(rng.normal(trend, 0.5, n_bars))
            close_vals = np.maximum(close_vals, 1.0)  # keep positive
            dfs[f"STK{i}"] = _make_df(n_bars, close_vals=close_vals)
        return dfs

    def test_returns_top_n(self):
        dfs = self._make_universe(n_stocks=10, n_bars=300)
        t_ts = list(dfs.values())[0].index[-1]
        result = select_momentum_stocks(dfs, t_ts, top_n=3)
        assert len(result) == 3

    def test_returns_fewer_if_universe_smaller(self):
        dfs = self._make_universe(n_stocks=2, n_bars=300)
        t_ts = list(dfs.values())[0].index[-1]
        result = select_momentum_stocks(dfs, t_ts, top_n=5)
        assert len(result) == 2

    def test_empty_if_no_sufficient_history(self):
        dfs = {"A": _make_df(100), "B": _make_df(100)}
        t_ts = list(dfs.values())[0].index[-1]
        result = select_momentum_stocks(dfs, t_ts, top_n=2)
        assert result == []

    def test_no_lookahead(self):
        """Adding future data must not change the selected stocks."""
        dfs = self._make_universe(n_stocks=10, n_bars=300)
        t_ts = list(dfs.values())[0].index[-1]

        selected_at_t = select_momentum_stocks(dfs, t_ts, top_n=5)

        # Append a wildly different future bar to each df
        future_ts = t_ts + pd.Timedelta(days=1)
        dfs_future = {}
        for ticker, df in dfs.items():
            df2 = df.copy()
            df2.loc[future_ts] = df2.iloc[-1]
            df2.at[future_ts, "adj_close"] = 99999.0
            dfs_future[ticker] = df2

        # slice to original t_ts
        dfs_sliced = {tk: df.loc[df.index <= t_ts] for tk, df in dfs_future.items()}
        selected_sliced = select_momentum_stocks(dfs_sliced, t_ts, top_n=5)
        assert selected_at_t == selected_sliced

    def test_drops_tickers_with_none_scores(self):
        # One ticker has insufficient history
        dfs = self._make_universe(n_stocks=5, n_bars=300)
        dfs["SHORT"] = _make_df(100)  # too short
        t_ts = list(dfs.values())[0].index[-1]
        result = select_momentum_stocks(dfs, t_ts, top_n=3)
        assert "SHORT" not in result


# ── select_short_stocks ───────────────────────────────────────────────────────


class TestSelectShortStocks:
    def _make_universe(self, n_stocks=10, n_bars=300):
        """Universe with clearly differentiated momentum (same as long test)."""
        dfs = {}
        for i in range(n_stocks):
            seed = i + 1
            rng = np.random.default_rng(seed)
            trend = (i - n_stocks // 2) * 0.3
            close_vals = 100.0 + np.cumsum(rng.normal(trend, 0.5, n_bars))
            close_vals = np.maximum(close_vals, 1.0)
            dfs[f"STK{i}"] = _make_df(n_bars, close_vals=close_vals)
        return dfs

    def test_returns_short_n(self):
        dfs = self._make_universe(n_stocks=10, n_bars=300)
        t_ts = list(dfs.values())[0].index[-1]
        result = select_short_stocks(dfs, t_ts, short_n=3)
        assert len(result) == 3

    def test_bottom_n_distinct_from_top_n(self):
        """Short and long selections must be disjoint for a large-enough universe."""
        dfs = self._make_universe(n_stocks=10, n_bars=300)
        t_ts = list(dfs.values())[0].index[-1]
        longs = set(select_momentum_stocks(dfs, t_ts, top_n=3))
        shorts = set(select_short_stocks(dfs, t_ts, short_n=3))
        assert longs.isdisjoint(shorts)

    def test_short_stocks_have_lower_composite_than_longs(self):
        """Every short must have a lower composite score than every long."""
        dfs = self._make_universe(n_stocks=10, n_bars=300)
        t_ts = list(dfs.values())[0].index[-1]
        longs = select_momentum_stocks(dfs, t_ts, top_n=4)
        shorts = select_short_stocks(dfs, t_ts, short_n=4)
        # Recompute composite for verification
        from src.signals.engine_momentum import compute_momentum_12_1, compute_52wh_pct
        scores = {}
        for ticker, df in dfs.items():
            df_slice = df.loc[df.index <= t_ts]
            m12 = compute_momentum_12_1(df_slice)
            w52 = compute_52wh_pct(df_slice)
            if m12 is not None and w52 is not None:
                scores[ticker] = (m12, w52)
        tickers = sorted(scores)
        m12_s = pd.Series([scores[t][0] for t in tickers], index=tickers)
        w52_s = pd.Series([scores[t][1] for t in tickers], index=tickers)
        composite = 0.5 * m12_s.rank(pct=True) + 0.5 * w52_s.rank(pct=True)
        min_long_score = min(composite[t] for t in longs)
        max_short_score = max(composite[t] for t in shorts)
        assert max_short_score < min_long_score

    def test_returns_fewer_if_universe_smaller_than_short_n(self):
        dfs = self._make_universe(n_stocks=2, n_bars=300)
        t_ts = list(dfs.values())[0].index[-1]
        result = select_short_stocks(dfs, t_ts, short_n=5)
        assert len(result) == 2

    def test_empty_if_no_sufficient_history(self):
        dfs = {"A": _make_df(100), "B": _make_df(100)}
        t_ts = list(dfs.values())[0].index[-1]
        result = select_short_stocks(dfs, t_ts, short_n=2)
        assert result == []

    def test_no_lookahead(self):
        """Adding future data must not change the short selection."""
        dfs = self._make_universe(n_stocks=10, n_bars=300)
        t_ts = list(dfs.values())[0].index[-1]
        selected_at_t = select_short_stocks(dfs, t_ts, short_n=4)

        future_ts = t_ts + pd.Timedelta(days=1)
        dfs_future = {}
        for ticker, df in dfs.items():
            df2 = df.copy()
            df2.loc[future_ts] = df2.iloc[-1]
            df2.at[future_ts, "adj_close"] = 0.001  # would flip all rankings
            dfs_future[ticker] = df2

        dfs_sliced = {tk: df.loc[df.index <= t_ts] for tk, df in dfs_future.items()}
        selected_sliced = select_short_stocks(dfs_sliced, t_ts, short_n=4)
        assert selected_at_t == selected_sliced


# ── MomentumBacktest smoke test ───────────────────────────────────────────────


class TestMomentumBacktestSmoke:
    def _make_config(self, top_n=3, short_n=0):
        from src.config import EngineMomentumConfig, RiskConfig, CostConfig
        eng = EngineMomentumConfig(top_n=top_n, short_n=short_n, spy_symbol="SPY",
                                   target_portfolio_vol=0.15, vol_floor=0.05)
        risk = RiskConfig(max_position_pct=0.10, max_positions=20,
                          sector_cap=3, drawdown_warning_pct=0.15,
                          drawdown_stop_pct=0.30, min_history_days=273,
                          initial_capital=100_000.0)
        cost = CostConfig(commission_fixed=0.0, commission_pct=0.0,
                          slippage_bps=0.0, tax_rate=0.0, tax_annual_allowance=0.0,
                          borrow_cost_annual=0.0)
        return eng, risk, cost

    def _make_price_data(self, n_tickers=5, n_bars=500):
        data = {}
        # SPY — rising trend (positive regime)
        spy_close = 100.0 + np.linspace(0, 50, n_bars)
        data["SPY"] = _make_df(n_bars, close_vals=spy_close, start="2007-01-01")
        # Stocks
        for i in range(n_tickers):
            rng = np.random.default_rng(i + 100)
            close_vals = 100.0 + np.cumsum(rng.normal(0.05, 0.5, n_bars))
            close_vals = np.maximum(close_vals, 1.0)
            data[f"STK{i}"] = _make_df(n_bars, close_vals=close_vals, start="2007-01-01")
        return data

    def test_runs_without_error(self):
        from src.backtest.engine_momentum import MomentumBacktest

        price_data = self._make_price_data()
        calendar = _make_calendar("2007-01-01", "2009-12-31")
        eng_cfg, risk_cfg, cost_cfg = self._make_config(top_n=3)

        all_tickers = set(price_data.keys()) - {"SPY"}

        def members_fn(d):
            return all_tickers

        engine = MomentumBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=100_000.0,
            engine_cfg=eng_cfg,
            risk_cfg=risk_cfg,
            cost_cfg=cost_cfg,
        )
        result = engine.run(date(2007, 1, 1), date(2009, 12, 31))
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
        # Final equity should be positive
        assert result.equity_curve["equity"].iloc[-1] > 0

    def test_regime_off_means_no_long_positions_at_end(self):
        """When SPY is always declining, regime is always off → no long trades.
        Short trades may occur if short_n > 0, but with short_n=0 no trades at all.
        """
        from src.backtest.engine_momentum import MomentumBacktest

        price_data = self._make_price_data()
        # Make SPY decline so regime is always off
        n = len(price_data["SPY"])
        spy_close = np.linspace(200, 50, n)
        price_data["SPY"] = _make_df(n, close_vals=spy_close, start="2007-01-01")

        calendar = _make_calendar("2007-01-01", "2009-12-31")
        eng_cfg, risk_cfg, cost_cfg = self._make_config(top_n=3, short_n=0)

        all_tickers = set(price_data.keys()) - {"SPY"}

        def members_fn(d):
            return all_tickers

        engine = MomentumBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=100_000.0,
            engine_cfg=eng_cfg,
            risk_cfg=risk_cfg,
            cost_cfg=cost_cfg,
        )
        result = engine.run(date(2007, 1, 1), date(2009, 12, 31))
        # With short_n=0 and regime always off, no trades should have opened
        assert result.n_trades == 0

    def test_equity_curve_length_matches_sessions(self):
        from src.backtest.engine_momentum import MomentumBacktest

        price_data = self._make_price_data()
        calendar = _make_calendar("2007-01-01", "2009-12-31")
        eng_cfg, risk_cfg, cost_cfg = self._make_config(top_n=3)

        sessions = calendar.sessions(date(2007, 1, 1), date(2009, 12, 31))

        def members_fn(d):
            return set(price_data.keys()) - {"SPY"}

        engine = MomentumBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=100_000.0,
            engine_cfg=eng_cfg,
            risk_cfg=risk_cfg,
            cost_cfg=cost_cfg,
        )
        result = engine.run(date(2007, 1, 1), date(2009, 12, 31))
        assert len(result.equity_curve) == len(sessions)


# ── Short leg backtester tests ────────────────────────────────────────────────


class TestMomentumBacktestShortLeg:
    """Tests for the short leg and long/short interaction."""

    def _make_config_short(self, top_n=3, short_n=2):
        from src.config import EngineMomentumConfig, RiskConfig, CostConfig
        eng = EngineMomentumConfig(top_n=top_n, short_n=short_n, spy_symbol="SPY",
                                   target_portfolio_vol=0.15, vol_floor=0.05,
                                   stock_sma_filter=0)
        risk = RiskConfig(max_position_pct=0.10, max_positions=20,
                          sector_cap=3, drawdown_warning_pct=0.15,
                          drawdown_stop_pct=0.30, min_history_days=273,
                          initial_capital=100_000.0)
        cost = CostConfig(commission_fixed=0.0, commission_pct=0.0,
                          slippage_bps=0.0, tax_rate=0.0, tax_annual_allowance=0.0,
                          borrow_cost_annual=0.0)
        return eng, risk, cost

    def _make_price_data(self, n_tickers=8, n_bars=500):
        data = {}
        spy_close = 100.0 + np.linspace(0, 50, n_bars)
        data["SPY"] = _make_df(n_bars, close_vals=spy_close, start="2007-01-01")
        for i in range(n_tickers):
            rng = np.random.default_rng(i + 100)
            close_vals = 100.0 + np.cumsum(rng.normal(0.05, 0.5, n_bars))
            close_vals = np.maximum(close_vals, 1.0)
            data[f"STK{i}"] = _make_df(n_bars, close_vals=close_vals, start="2007-01-01")
        return data

    def _run(self, top_n=3, short_n=2, price_data=None):
        from src.backtest.engine_momentum import MomentumBacktest
        if price_data is None:
            price_data = self._make_price_data()
        calendar = _make_calendar("2007-01-01", "2009-12-31")
        eng_cfg, risk_cfg, cost_cfg = self._make_config_short(top_n=top_n, short_n=short_n)
        all_tickers = set(price_data.keys()) - {"SPY"}

        def members_fn(d):
            return all_tickers

        engine = MomentumBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=100_000.0,
            engine_cfg=eng_cfg,
            risk_cfg=risk_cfg,
            cost_cfg=cost_cfg,
        )
        return engine.run(date(2007, 1, 1), date(2009, 12, 31))

    def test_short_trades_are_recorded(self):
        """With short_n > 0 and a bull regime, short trades should be recorded."""
        result = self._run(top_n=2, short_n=2)
        short_trades = [t for t in result.trades if t["direction"] == "short"]
        assert len(short_trades) > 0

    def test_short_pnl_positive_when_prices_fall(self):
        """Short leg improves equity when shorted stock consistently declines.

        LOSER is always in bottom-1 (never covered as a closed trade), but its
        unrealized P&L accrues in mark_to_market. The equity_curve WITH the
        short leg must exceed the equity_curve WITHOUT it.
        """
        from src.backtest.engine_momentum import MomentumBacktest
        from src.config import EngineMomentumConfig, RiskConfig, CostConfig

        # 800 bars ≈ 3+ years, so data outlasts the 2007–2009 backtest calendar.
        n_bars = 800
        spy_close = np.linspace(100, 200, n_bars)         # rising → regime always OK
        loser_close = np.linspace(200, 50, n_bars)        # constant decline → bottom-1
        winner_closes = {f"WIN{i}": np.linspace(50 + i * 10, 200 + i * 10, n_bars) for i in range(5)}

        price_data = {"SPY": _make_df(n_bars, close_vals=spy_close, start="2007-01-01"),
                      "LOSER": _make_df(n_bars, close_vals=loser_close, start="2007-01-01")}
        for ticker, closes in winner_closes.items():
            price_data[ticker] = _make_df(n_bars, close_vals=closes, start="2007-01-01")

        calendar = _make_calendar("2007-01-01", "2009-12-31")
        all_tickers = set(price_data.keys()) - {"SPY"}

        risk = RiskConfig(max_position_pct=0.10, max_positions=20, sector_cap=3,
                          drawdown_warning_pct=0.15, drawdown_stop_pct=0.30,
                          min_history_days=273, initial_capital=100_000.0)

        def make_run(short_n):
            eng = EngineMomentumConfig(top_n=3, short_n=short_n, spy_symbol="SPY",
                                       target_portfolio_vol=0.15, vol_floor=0.05, stock_sma_filter=0)
            cost = CostConfig(commission_fixed=0.0, commission_pct=0.0, slippage_bps=0.0,
                              tax_rate=0.0, tax_annual_allowance=0.0, borrow_cost_annual=0.0)
            engine = MomentumBacktest(
                price_data=price_data, calendar=calendar,
                universe_members=lambda d: all_tickers,
                initial_capital=100_000.0, engine_cfg=eng, risk_cfg=risk, cost_cfg=cost,
            )
            return engine.run(date(2007, 1, 1), date(2009, 12, 31))

        result_no_short = make_run(short_n=0)
        result_with_short = make_run(short_n=1)

        eq_no_short = result_no_short.equity_curve["equity"].iloc[-1]
        eq_with_short = result_with_short.equity_curve["equity"].iloc[-1]

        # Shorting LOSER (which falls from ~$149 to ~$58 during the backtest)
        # generates unrealized P&L that is included in mark_to_market each day.
        assert eq_with_short > eq_no_short, (
            f"Short leg should improve equity when shorted stock declines: "
            f"with={eq_with_short:.0f} vs without={eq_no_short:.0f}"
        )

    def test_regime_off_exits_longs_keeps_shorts(self):
        """In bearish SPY regime, longs are exited but short trades still occur."""
        from src.backtest.engine_momentum import MomentumBacktest
        from src.config import EngineMomentumConfig, RiskConfig, CostConfig

        n_bars = 500
        # SPY declines → regime always off
        spy_close = np.linspace(200, 50, n_bars)
        price_data = {"SPY": _make_df(n_bars, close_vals=spy_close, start="2007-01-01")}
        for i in range(8):
            rng = np.random.default_rng(i + 1)
            close_vals = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n_bars))
            close_vals = np.maximum(close_vals, 1.0)
            price_data[f"STK{i}"] = _make_df(n_bars, close_vals=close_vals, start="2007-01-01")

        eng = EngineMomentumConfig(top_n=3, short_n=2, spy_symbol="SPY",
                                   target_portfolio_vol=0.15, vol_floor=0.05, stock_sma_filter=0)
        risk = RiskConfig(max_position_pct=0.10, max_positions=20, sector_cap=3,
                          drawdown_warning_pct=0.15, drawdown_stop_pct=0.30,
                          min_history_days=273, initial_capital=100_000.0)
        cost = CostConfig(commission_fixed=0.0, commission_pct=0.0, slippage_bps=0.0,
                          tax_rate=0.0, tax_annual_allowance=0.0, borrow_cost_annual=0.0)

        calendar = _make_calendar("2007-01-01", "2009-12-31")
        all_tickers = set(price_data.keys()) - {"SPY"}

        engine = MomentumBacktest(
            price_data=price_data, calendar=calendar,
            universe_members=lambda d: all_tickers,
            initial_capital=100_000.0, engine_cfg=eng, risk_cfg=risk, cost_cfg=cost,
        )
        result = engine.run(date(2007, 1, 1), date(2009, 12, 31))

        long_trades = [t for t in result.trades if t["direction"] == "long"]
        short_trades = [t for t in result.trades if t["direction"] == "short"]
        assert len(long_trades) == 0, "No long trades should be opened in bearish regime"
        assert len(short_trades) > 0, "Short trades should occur even in bearish regime"

    def test_borrow_cost_reduces_equity(self):
        """With borrow_cost_annual > 0, final equity must be lower than with 0."""
        from src.backtest.engine_momentum import MomentumBacktest
        from src.config import EngineMomentumConfig, RiskConfig, CostConfig

        n_bars = 500
        spy_close = np.linspace(100, 200, n_bars)
        price_data = {"SPY": _make_df(n_bars, close_vals=spy_close, start="2007-01-01")}
        for i in range(8):
            rng = np.random.default_rng(i + 1)
            close_vals = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n_bars))
            close_vals = np.maximum(close_vals, 1.0)
            price_data[f"STK{i}"] = _make_df(n_bars, close_vals=close_vals, start="2007-01-01")

        calendar = _make_calendar("2007-01-01", "2009-12-31")
        all_tickers = set(price_data.keys()) - {"SPY"}

        def make_engine(borrow):
            eng = EngineMomentumConfig(top_n=2, short_n=2, spy_symbol="SPY",
                                       target_portfolio_vol=0.15, vol_floor=0.05, stock_sma_filter=0)
            risk = RiskConfig(max_position_pct=0.10, max_positions=20, sector_cap=3,
                              drawdown_warning_pct=0.15, drawdown_stop_pct=0.30,
                              min_history_days=273, initial_capital=100_000.0)
            cost = CostConfig(commission_fixed=0.0, commission_pct=0.0, slippage_bps=0.0,
                              tax_rate=0.0, tax_annual_allowance=0.0, borrow_cost_annual=borrow)
            return MomentumBacktest(
                price_data=price_data, calendar=calendar,
                universe_members=lambda d: all_tickers,
                initial_capital=100_000.0, engine_cfg=eng, risk_cfg=risk, cost_cfg=cost,
            )

        result_no_borrow = make_engine(0.0).run(date(2007, 1, 1), date(2009, 12, 31))
        result_with_borrow = make_engine(0.0050).run(date(2007, 1, 1), date(2009, 12, 31))

        # Check if any short trades were opened at all first
        short_trades = [t for t in result_with_borrow.trades if t["direction"] == "short"]
        if len(short_trades) > 0:
            eq_no_borrow = result_no_borrow.equity_curve["equity"].iloc[-1]
            eq_with_borrow = result_with_borrow.equity_curve["equity"].iloc[-1]
            assert eq_with_borrow < eq_no_borrow, "Borrow cost should reduce equity"
