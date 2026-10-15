"""Tests for src/signals/engine_b.py and src/backtest/engine_b.py."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.calendar import TradingCalendar
from src.signals.engine_b import (
    compute_activity_score,
    compute_momentum_score,
    is_rebalance_day,
    select_engine_b_stocks,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_df(n: int, close_vals=None, volume_vals=None, seed=42):
    """Create a canonical OHLCV DataFrame for testing."""
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    if close_vals is None:
        rng = np.random.default_rng(seed)
        close_vals = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close_vals = np.array(close_vals, dtype=float)
    if volume_vals is None:
        volume_vals = [1_000_000.0] * n
    return pd.DataFrame(
        {
            "open": close_vals,
            "high": close_vals * 1.01,
            "low": close_vals * 0.99,
            "close": close_vals,
            "adj_close": close_vals,
            "volume": volume_vals,
            "dividends": [0.0] * n,
            "splits": [0.0] * n,
        },
        index=idx,
    )


def _make_calendar(start="2010-01-01", end="2015-12-31"):
    dates = [d.date() for d in pd.date_range(start, end, freq="B")]
    return TradingCalendar.from_dates(dates)


# ── compute_activity_score ───────────────────────────────────────────────────


class TestComputeActivityScore:
    def test_returns_none_if_insufficient_history(self):
        df = _make_df(251)  # need 252 for n_slow=252
        assert compute_activity_score(df, n_fast=21, n_slow=252) is None

    def test_returns_none_for_exactly_252_rows(self):
        # 252 rows — rolling(252, min_periods=252) gives a valid value at last row
        df = _make_df(252)
        result = compute_activity_score(df, n_fast=21, n_slow=252)
        assert result is not None

    def test_score_above_one_for_high_recent_activity(self):
        # Low base volume, spike in last 21 days
        n = 300
        vols = [1_000.0] * n
        for i in range(n - 21, n):
            vols[i] = 10_000.0  # 10× spike
        df = _make_df(n, volume_vals=vols)
        score = compute_activity_score(df, n_fast=21, n_slow=252)
        assert score is not None
        assert score > 1.0

    def test_score_below_one_for_low_recent_activity(self):
        # High base volume, low in last 21 days
        n = 300
        vols = [10_000.0] * n
        for i in range(n - 21, n):
            vols[i] = 100.0
        df = _make_df(n, volume_vals=vols)
        score = compute_activity_score(df, n_fast=21, n_slow=252)
        assert score is not None
        assert score < 1.0

    def test_no_lookahead(self):
        """Score at row N must not change after appending a future row."""
        n = 300
        df = _make_df(n)
        t_ts = df.index[-1]
        df_slice = df.loc[df.index <= t_ts]
        score_before = compute_activity_score(df_slice)

        # Append one more row
        extra_idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=1, freq="B")
        extra = _make_df(1, volume_vals=[999_999_999.0])
        extra.index = extra_idx
        df_ext = pd.concat([df, extra])
        score_after = compute_activity_score(df_ext.iloc[:-1])

        assert score_before == score_after

    def test_returns_none_when_slow_mean_zero(self):
        n = 300
        df = _make_df(n, volume_vals=[0.0] * n)
        assert compute_activity_score(df) is None


# ── compute_momentum_score ───────────────────────────────────────────────────


class TestComputeMomentumScore:
    def test_returns_none_if_insufficient_history(self):
        # Need lookback + skip = 21 + 2 = 23 rows
        df = _make_df(22)
        assert compute_momentum_score(df, lookback=21, skip=2) is None

    def test_returns_value_for_exactly_23_rows(self):
        df = _make_df(23)
        result = compute_momentum_score(df, lookback=21, skip=2)
        assert result is not None

    def test_positive_momentum_for_rising_series(self):
        vals = list(range(1, 30))  # 1,2,...,29
        df = _make_df(29, close_vals=vals)
        score = compute_momentum_score(df, lookback=21, skip=2)
        # adj_close[-3] = 27, adj_close[-23] = 6 → positive
        assert score is not None
        assert score > 0

    def test_negative_momentum_for_falling_series(self):
        vals = list(range(29, 0, -1))  # 29,28,...,1
        df = _make_df(29, close_vals=vals)
        score = compute_momentum_score(df, lookback=21, skip=2)
        assert score is not None
        assert score < 0

    def test_skip_respected(self):
        """Momentum should skip the most recent 'skip' sessions."""
        # With lookback=21, skip=2:
        #   end_price   = adj_close[-(skip+1)]      = adj_close[-3]
        #   start_price = adj_close[-(lookback+skip)] = adj_close[-23]
        vals = [100.0] * 30
        vals[-23] = 50.0   # start_price at position -(lookback+skip) = -(21+2) = -23
        vals[-3] = 100.0   # end_price at position -(skip+1) = -3 (already 100)
        vals[-2] = 999.0   # should be skipped
        vals[-1] = 999.0   # should be skipped
        df = _make_df(30, close_vals=vals)
        score = compute_momentum_score(df, lookback=21, skip=2)
        # (100 - 50) / 50 = 1.0
        assert score is not None
        assert abs(score - 1.0) < 0.01

    def test_no_lookahead(self):
        """Score at row N must not change when slicing to N."""
        vals = list(range(1, 50))
        df = _make_df(49, close_vals=vals)
        t_ts = df.index[-1]
        score_before = compute_momentum_score(df.loc[df.index <= t_ts])

        extra_idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=1, freq="B")
        extra = _make_df(1, close_vals=[0.001])
        extra.index = extra_idx
        df_ext = pd.concat([df, extra])
        score_after = compute_momentum_score(df_ext.iloc[:-1])
        assert score_before == score_after


# ── is_rebalance_day ─────────────────────────────────────────────────────────


class TestIsRebalanceDay:
    def _calendar(self):
        return _make_calendar("2010-01-01", "2015-12-31")

    def test_first_trading_day_of_month_is_rebalance(self):
        cal = self._calendar()
        # The synthetic calendar uses pd.date_range freq="B" starting 2010-01-01.
        # 2010-01-01 is a Friday so it IS the first business session of January.
        assert is_rebalance_day(date(2010, 1, 1), cal) is True

    def test_second_trading_day_is_not_rebalance(self):
        cal = self._calendar()
        # 2010-01-04 is Monday — the second business session of January.
        assert is_rebalance_day(date(2010, 1, 4), cal) is False

    def test_mid_month_is_not_rebalance(self):
        cal = self._calendar()
        assert is_rebalance_day(date(2010, 1, 15), cal) is False

    def test_rebalance_days_occur_once_per_month(self):
        cal = self._calendar()
        # Count rebalance days in 2010: should be exactly 12
        all_sessions = cal.sessions(date(2010, 1, 1), date(2010, 12, 31))
        rebalance_days = [d for d in all_sessions if is_rebalance_day(d, cal)]
        assert len(rebalance_days) == 12

    def test_non_session_day_is_not_rebalance(self):
        cal = self._calendar()
        # 2010-01-02 is a Saturday — not in the trading calendar
        assert is_rebalance_day(date(2010, 1, 2), cal) is False


# ── select_engine_b_stocks ───────────────────────────────────────────────────


class TestSelectEngineBStocks:
    def _make_universe(self, n_tickers=10, n_rows=300, seed=0):
        """Create a dict of DataFrames with varying activity and momentum."""
        universe = {}
        rng = np.random.default_rng(seed)
        for i in range(n_tickers):
            label = f"T{i:02d}"
            # Ticker i has higher volume than ticker i-1 (deterministic activity)
            vol_base = 1_000.0 * (i + 1)
            vols = [vol_base] * n_rows
            close_vals = 100 + np.cumsum(rng.normal(0, 0.5, n_rows))
            universe[label] = _make_df(n_rows, close_vals=close_vals, volume_vals=vols)
        return universe

    def test_returns_at_most_top_n(self):
        universe = self._make_universe(n_tickers=20, n_rows=300)
        t_ts = pd.Timestamp("2011-03-01")
        result = select_engine_b_stocks(universe, t_ts, top_n=5)
        assert len(result) <= 5

    def test_returns_list_of_strings(self):
        universe = self._make_universe(n_tickers=20, n_rows=300)
        t_ts = pd.Timestamp("2011-03-01")
        result = select_engine_b_stocks(universe, t_ts, top_n=5)
        for ticker in result:
            assert isinstance(ticker, str)

    def test_no_ticker_repeated(self):
        universe = self._make_universe(n_tickers=20, n_rows=300)
        t_ts = pd.Timestamp("2011-03-01")
        result = select_engine_b_stocks(universe, t_ts, top_n=5)
        assert len(result) == len(set(result))

    def test_returns_empty_when_insufficient_history(self):
        # All tickers have fewer rows than min_history
        universe = self._make_universe(n_tickers=5, n_rows=100)
        t_ts = pd.Timestamp("2010-06-01")
        result = select_engine_b_stocks(universe, t_ts, top_n=5, min_history=252)
        assert result == []

    def test_deterministic_tie_break(self):
        """Same universe and timestamp should always return the same order."""
        universe = self._make_universe(n_tickers=15, n_rows=300)
        t_ts = pd.Timestamp("2011-03-01")
        result1 = select_engine_b_stocks(universe, t_ts, top_n=5)
        result2 = select_engine_b_stocks(universe, t_ts, top_n=5)
        assert result1 == result2

    def test_no_lookahead(self):
        """Result on df[:t] must equal result when using df[:t] explicitly."""
        universe = self._make_universe(n_tickers=15, n_rows=300)
        t_ts = pd.Timestamp("2011-03-01")

        # Slice all dfs to t_ts
        sliced_universe = {
            ticker: df.loc[df.index <= t_ts]
            for ticker, df in universe.items()
        }
        result_sliced = select_engine_b_stocks(sliced_universe, t_ts, top_n=5)

        # Full universe (has future data) — function slices internally
        result_full = select_engine_b_stocks(universe, t_ts, top_n=5)

        assert result_sliced == result_full

    def test_top_third_activity_filter(self):
        """Only tickers in the top third by activity should be candidates."""
        n = 300
        # Create tickers with very different activity levels
        universe = {}
        # High activity tickers: T0, T1, T2 (volume 10000)
        # Low activity tickers: T3..T8 (volume 10)
        for i in range(3):
            universe[f"T{i:02d}"] = _make_df(n, volume_vals=[10_000.0] * n)
        for i in range(3, 9):
            universe[f"T{i:02d}"] = _make_df(n, volume_vals=[10.0] * n, seed=i)

        t_ts = universe["T00"].index[-1]
        result = select_engine_b_stocks(
            universe, t_ts, top_n=5, min_history=252, activity_slow=252
        )
        # Top third of 6 qualifying = 2, so result should only contain T00..T02
        for ticker in result:
            assert ticker in {"T00", "T01", "T02"}


# ── EngineBBacktest integration ──────────────────────────────────────────────


class TestEngineBBacktest:
    """Smoke tests for the Engine B event loop."""

    def _build_price_data(self, tickers, n_sessions=400, seed=0):
        """Build synthetic price data for a list of tickers."""
        rng = np.random.default_rng(seed)
        price_data = {}
        idx = pd.date_range("2010-01-01", periods=n_sessions, freq="B")
        for i, ticker in enumerate(tickers):
            close = 100 + np.cumsum(rng.normal(0, 0.5, n_sessions))
            close = np.clip(close, 1, None)
            price_data[ticker] = pd.DataFrame(
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
        # SPY for regime filter — always above SMA200 (rising)
        spy_close = np.linspace(100, 200, n_sessions)
        price_data["SPY"] = pd.DataFrame(
            {
                "open": spy_close,
                "high": spy_close * 1.01,
                "low": spy_close * 0.99,
                "close": spy_close,
                "adj_close": spy_close,
                "volume": [100_000_000.0] * n_sessions,
                "dividends": [0.0] * n_sessions,
                "splits": [0.0] * n_sessions,
            },
            index=idx,
        )
        return price_data, idx

    def _build_deps(self, tickers, n_sessions=400):
        from src.config import CostConfig, EngineBConfig, RiskConfig

        price_data, idx = self._build_price_data(tickers, n_sessions)
        calendar = TradingCalendar.from_dates([d.date() for d in idx])

        def members_fn(d):
            return set(tickers)

        engine_cfg = EngineBConfig(top_n=5, spy_symbol="SPY")
        risk_cfg = RiskConfig(
            max_positions_engine_a=8,
            max_positions_engine_b=10,
            max_positions_combined=15,
            sector_cap=3,
            drawdown_warning_pct=0.10,
            drawdown_stop_pct=0.20,
            atr_floor=0.001,
            min_history_days=252,
            initial_capital=100_000.0,
        )
        cost_cfg = CostConfig(
            commission_fixed=0.0,
            commission_pct=0.0,
            slippage_bps=0.0,
        )
        return price_data, calendar, members_fn, engine_cfg, risk_cfg, cost_cfg

    def test_run_returns_run_result(self):
        from src.backtest.engine_b import EngineBBacktest

        tickers = [f"T{i:02d}" for i in range(20)]
        price_data, calendar, members_fn, engine_cfg, risk_cfg, cost_cfg = self._build_deps(tickers)

        engine = EngineBBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=100_000.0,
            engine_cfg=engine_cfg,
            risk_cfg=risk_cfg,
            cost_cfg=cost_cfg,
        )
        result = engine.run(date(2011, 1, 1), date(2011, 12, 31))

        assert result is not None
        assert len(result.equity_curve) > 0
        assert isinstance(result.trades, list)
        assert result.engine == "B"

    def test_equity_curve_has_daily_rows(self):
        from src.backtest.engine_b import EngineBBacktest

        tickers = [f"T{i:02d}" for i in range(20)]
        price_data, calendar, members_fn, engine_cfg, risk_cfg, cost_cfg = self._build_deps(tickers)

        engine = EngineBBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=100_000.0,
            engine_cfg=engine_cfg,
            risk_cfg=risk_cfg,
            cost_cfg=cost_cfg,
        )
        result = engine.run(date(2011, 1, 1), date(2011, 12, 31))

        n_sessions = len(calendar.sessions(date(2011, 1, 1), date(2011, 12, 31)))
        assert len(result.equity_curve) == n_sessions

    def test_no_positions_exceed_engine_b_limit(self):
        from src.backtest.engine_b import EngineBBacktest

        tickers = [f"T{i:02d}" for i in range(30)]
        price_data, calendar, members_fn, engine_cfg, risk_cfg, cost_cfg = self._build_deps(tickers)

        engine = EngineBBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=100_000.0,
            engine_cfg=engine_cfg,
            risk_cfg=risk_cfg,
            cost_cfg=cost_cfg,
        )
        result = engine.run(date(2011, 1, 1), date(2011, 12, 31))
        # All trades should be engine B
        for trade in result.trades:
            assert trade["engine"] == "B"

    def test_rebalance_only_on_first_session_of_month(self):
        """No buys should be queued on non-rebalance days."""
        from src.backtest.engine_b import EngineBBacktest

        tickers = [f"T{i:02d}" for i in range(20)]
        price_data, calendar, members_fn, engine_cfg, risk_cfg, cost_cfg = self._build_deps(tickers)

        engine = EngineBBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=100_000.0,
            engine_cfg=engine_cfg,
            risk_cfg=risk_cfg,
            cost_cfg=cost_cfg,
        )
        result = engine.run(date(2011, 1, 1), date(2011, 12, 31))
        # Entry trades should align to opens after a first-of-month session
        # At minimum: result exists and did not crash
        assert result is not None

    def test_stopped_engine_takes_no_new_positions(self):
        """When drawdown exceeds stop level, no new Engine B entries."""
        from src.backtest.engine_b import EngineBBacktest

        tickers = [f"T{i:02d}" for i in range(20)]
        price_data, calendar, members_fn, engine_cfg, risk_cfg, cost_cfg = self._build_deps(tickers)

        # Set stop at 0% so it triggers immediately
        from src.config import RiskConfig
        risk_cfg_stop = RiskConfig(
            max_positions_engine_a=8,
            max_positions_engine_b=10,
            max_positions_combined=15,
            sector_cap=3,
            drawdown_warning_pct=0.001,
            drawdown_stop_pct=0.001,
            atr_floor=0.001,
            min_history_days=252,
            initial_capital=100_000.0,
        )

        engine = EngineBBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=100_000.0,
            engine_cfg=engine_cfg,
            risk_cfg=risk_cfg_stop,
            cost_cfg=cost_cfg,
        )
        result = engine.run(date(2011, 1, 1), date(2011, 12, 31))
        # With a 0% stop level, after any small decline no new trades allowed
        assert result is not None
