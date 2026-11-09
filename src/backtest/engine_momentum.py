"""Momentum Engine monthly-rebalance backtester (long/short).

Event loop (each session t):
  1. Process fills  — pending orders filled at t's open
  2. Collect close prices for all open positions
  3. Accrue daily borrow cost for short positions
  4. Mark-to-market — update peak equity, check drawdown
  5. Check regime (SPY 12-1 momentum + SMA200)
  6. Asymmetric regime logic:
       Regime bad OR drawdown-stopped:
         → queue sells for all LONG positions (exit longs)
         → SHORT book is kept (bearish = good for shorts)
       Regime OK AND not stopped:
         → rebalance longs on first session of month
  7. Rebalance SHORT book on first session of month (always, regardless of regime)
  8. Year-end tax
  9. Log equity

Short leg behaviour (Option A — Asymmetric):
  - Bull regime: hold long top_n + short bottom short_n
  - Bear regime: exit all longs, keep/open short bottom short_n
  - Drawdown stop: exit ALL positions (both sides)

No mid-month exits outside rebalance/regime/drawdown-stop.
No look-ahead: all signals use df.loc[:t_ts].
Fills use the next session's open price (t+1).
"""
from __future__ import annotations

import logging
import math
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from ..data.calendar import TradingCalendar
from ..portfolio.positions import OrderSide, PendingOrder, Position, PositionBook
from ..portfolio.risk import RiskManager, SectorMap
from ..portfolio.sizing import compute_shares_vol_target
from ..signals.engine_momentum import (
    compute_spy_regime,
    compute_stock_above_sma,
    is_rebalance_day,
    select_momentum_stocks,
    select_short_stocks,
)
from .engine import RunResult
from .fills import compute_borrow_cost, compute_commission, compute_fill_price
from .tax import GermanTaxLedger

logger = logging.getLogger(__name__)


def _stock_annual_vol(df_slice: pd.DataFrame, window: int = 21) -> float:
    """Rolling 21-day std of daily log-returns × sqrt(252)."""
    closes = df_slice["adj_close"]
    if len(closes) < window + 1:
        return 0.0
    log_rets = np.log(closes / closes.shift(1)).dropna()
    if len(log_rets) < window:
        return 0.0
    vol = float(log_rets.iloc[-window:].std())
    return vol * math.sqrt(252)


class MomentumBacktest:
    """Monthly-rebalance cross-sectional momentum backtester (long/short).

    Parameters
    ----------
    price_data      : dict[ticker, DataFrame] — canonical OHLCV, indexed by tz-naive Timestamp
    calendar        : TradingCalendar
    universe_members: callable(date) -> set[str]
    initial_capital : starting cash in USD
    engine_cfg      : EngineMomentumConfig
    risk_cfg        : RiskConfig
    cost_cfg        : CostConfig
    stress          : if True, use stress_cost_multiplier for all commissions/slippage
    sector_map      : optional SectorMap
    """

    def __init__(
        self,
        price_data: dict,
        calendar: TradingCalendar,
        universe_members,
        initial_capital: float,
        engine_cfg,
        risk_cfg,
        cost_cfg,
        stress: bool = False,
        sector_map: Optional[SectorMap] = None,
    ) -> None:
        self._data = price_data
        self._calendar = calendar
        self._members_fn = universe_members
        self._initial_capital = initial_capital
        self._eng = engine_cfg
        self._risk = risk_cfg
        self._cost = cost_cfg
        self._stress = stress
        self._multiplier = cost_cfg.stress_cost_multiplier if stress else 1.0
        self._sector_map = sector_map or SectorMap.empty()

    # ── main entry ─────────────────────────────────────────────────────────────

    def run(self, start_date: date, end_date: date) -> RunResult:
        sessions = self._calendar.sessions(start_date, end_date)
        if not sessions:
            raise ValueError(f"No sessions in [{start_date}, {end_date}]")

        cash = float(self._initial_capital)
        book = PositionBook()
        tax_ledger = GermanTaxLedger(
            tax_rate=self._cost.tax_rate,
            annual_allowance_eur=self._cost.tax_annual_allowance,
        )
        risk_mgr = RiskManager(
            max_positions=self._risk.max_positions,
            sector_cap=self._risk.sector_cap,
            drawdown_warning_pct=self._risk.drawdown_warning_pct,
            drawdown_stop_pct=self._risk.drawdown_stop_pct,
            initial_equity=cash,
            sector_map=self._sector_map,
        )

        equity_rows: list[tuple[date, float]] = []
        trades: list[dict] = []
        total_comm = 0.0
        total_tax = 0.0
        total_slip = 0.0
        total_borrow = 0.0

        short_n = getattr(self._eng, "short_n", 0)
        borrow_rate = getattr(self._cost, "borrow_cost_annual", 0.0)

        for i, t in enumerate(sessions):
            t_ts = pd.Timestamp(t)
            next_session = sessions[i + 1] if i + 1 < len(sessions) else None

            # ── 1. Process fills ──────────────────────────────────────────────
            orders = book.flush_pending()
            for order in orders:
                if order.shares == 0:
                    continue
                ticker = order.ticker
                df_ticker = self._data.get(ticker)
                if df_ticker is None or df_ticker.empty:
                    continue

                row = df_ticker.loc[df_ticker.index <= t_ts]
                if row.empty or row.index[-1].date() != t:
                    continue
                raw_open = float(row["open"].iloc[-1])

                if order.direction == "long":
                    if order.side == OrderSide.BUY:
                        # Long entry
                        fill = compute_fill_price(raw_open, "buy", self._cost.slippage_bps, self._multiplier)
                        trade_val = order.shares * fill
                        comm = compute_commission(self._cost.commission_fixed, self._cost.commission_pct, trade_val, self._multiplier)
                        slip_cost = order.shares * (fill - raw_open)
                        cash -= trade_val + comm
                        total_comm += comm
                        total_slip += abs(slip_cost)

                        sector = self._sector_map.get(ticker)
                        pos = Position(
                            ticker=ticker,
                            shares=order.shares,
                            entry_date=t,
                            entry_price=fill,
                            entry_cost_usd=comm,
                            engine=order.engine,
                            sector=sector,
                            direction="long",
                        )
                        try:
                            book.open_position(pos)
                        except ValueError:
                            cash += trade_val + comm
                            total_comm -= comm
                            total_slip -= abs(slip_cost)

                    else:  # SELL — long exit
                        pos = book.get(ticker)
                        if pos is None:
                            continue
                        fill = compute_fill_price(raw_open, "sell", self._cost.slippage_bps, self._multiplier)
                        trade_val = pos.shares * fill
                        comm = compute_commission(self._cost.commission_fixed, self._cost.commission_pct, trade_val, self._multiplier)
                        slip_cost = pos.shares * (raw_open - fill)
                        cash += trade_val - comm
                        total_comm += comm
                        total_slip += abs(slip_cost)

                        realized_pnl = (fill - pos.entry_price) * pos.shares - pos.entry_cost_usd - comm
                        tax_ledger.record_trade(realized_pnl)
                        trades.append({
                            "ticker": ticker,
                            "engine": pos.engine,
                            "direction": "long",
                            "entry_date": pos.entry_date,
                            "exit_date": t,
                            "entry_price": pos.entry_price,
                            "exit_price": fill,
                            "shares": pos.shares,
                            "pnl_usd": realized_pnl,
                            "exit_reason": order.reason,
                            "sessions_held": len(self._calendar.sessions(pos.entry_date, t)),
                        })
                        book.close_position(ticker)

                else:  # direction == "short"
                    if order.side == OrderSide.SELL:
                        # Short entry: sell short, receive proceeds
                        fill = compute_fill_price(raw_open, "sell", self._cost.slippage_bps, self._multiplier)
                        trade_val = order.shares * fill
                        comm = compute_commission(self._cost.commission_fixed, self._cost.commission_pct, trade_val, self._multiplier)
                        slip_cost = order.shares * (raw_open - fill)
                        cash += trade_val - comm  # receive proceeds minus commission
                        total_comm += comm
                        total_slip += abs(slip_cost)

                        sector = self._sector_map.get(ticker)
                        pos = Position(
                            ticker=ticker,
                            shares=order.shares,
                            entry_date=t,
                            entry_price=fill,
                            entry_cost_usd=comm,
                            engine=order.engine,
                            sector=sector,
                            direction="short",
                        )
                        try:
                            book.open_short_position(pos)
                        except ValueError:
                            cash -= trade_val - comm
                            total_comm -= comm
                            total_slip -= abs(slip_cost)

                    else:  # BUY — short cover (exit short)
                        pos = book.get_short(ticker)
                        if pos is None:
                            continue
                        fill = compute_fill_price(raw_open, "buy", self._cost.slippage_bps, self._multiplier)
                        trade_val = pos.shares * fill
                        comm = compute_commission(self._cost.commission_fixed, self._cost.commission_pct, trade_val, self._multiplier)
                        slip_cost = pos.shares * (fill - raw_open)
                        cash -= trade_val + comm  # pay to cover + commission
                        total_comm += comm
                        total_slip += abs(slip_cost)

                        # Short P&L: profit when price falls
                        realized_pnl = (pos.entry_price - fill) * pos.shares - pos.entry_cost_usd - comm
                        tax_ledger.record_trade(realized_pnl)
                        trades.append({
                            "ticker": ticker,
                            "engine": pos.engine,
                            "direction": "short",
                            "entry_date": pos.entry_date,
                            "exit_date": t,
                            "entry_price": pos.entry_price,
                            "exit_price": fill,
                            "shares": pos.shares,
                            "pnl_usd": realized_pnl,
                            "exit_reason": order.reason,
                            "sessions_held": len(self._calendar.sessions(pos.entry_date, t)),
                        })
                        book.close_short_position(ticker)

            # ── 2. Collect close prices ───────────────────────────────────────
            all_tickers = book.open_tickers | book.short_tickers
            close_prices: dict[str, float] = {}
            for ticker in all_tickers:
                df_t = self._data.get(ticker)
                if df_t is not None:
                    row = df_t.loc[df_t.index <= t_ts]
                    if not row.empty and row.index[-1].date() == t:
                        close_prices[ticker] = float(row["adj_close"].iloc[-1])

            # ── 3. Accrue daily borrow cost for short positions ───────────────
            if borrow_rate > 0:
                for pos in book.short_positions():
                    price = close_prices.get(pos.ticker, pos.entry_price)
                    daily_borrow = compute_borrow_cost(pos.shares * price, borrow_rate)
                    cash -= daily_borrow
                    total_borrow += daily_borrow

            # ── 4. Mark-to-market equity ──────────────────────────────────────
            mtm_equity = cash + book.mark_to_market(close_prices)
            risk_mgr.update_equity(mtm_equity)
            dd_status = risk_mgr.check_drawdown(mtm_equity)
            if dd_status == "warning":
                logger.debug("%s drawdown warning: %.1f%%", t, risk_mgr.drawdown_from_peak(mtm_equity) * 100)

            # ── 5. Regime check ───────────────────────────────────────────────
            spy_symbol = self._eng.spy_symbol
            spy_df = self._data.get(spy_symbol)
            regime_ok = False
            if spy_df is not None:
                spy_slice = spy_df.loc[spy_df.index <= t_ts]
                regime_ok = compute_spy_regime(spy_slice)

            # ── 6. Drawdown stop → exit ALL positions (both sides) ────────────
            if risk_mgr.is_stopped and next_session is not None:
                for pos in book.long_positions():
                    book.add_pending(PendingOrder(pos.ticker, OrderSide.SELL, pos.shares, "momentum", "drawdown_stop", "long"))
                for pos in book.short_positions():
                    book.add_pending(PendingOrder(pos.ticker, OrderSide.BUY, pos.shares, "momentum", "drawdown_stop", "short"))

            # ── 7. Regime bad → exit longs only; shorts remain ────────────────
            elif not regime_ok and next_session is not None:
                for pos in book.long_positions():
                    book.add_pending(PendingOrder(pos.ticker, OrderSide.SELL, pos.shares, "momentum", "regime_exit", "long"))
                # Short book stays — bearish regime benefits shorts

            # ── 8. Rebalance long book (only when regime OK and not stopped) ──
            if (
                regime_ok
                and not risk_mgr.is_stopped
                and is_rebalance_day(t, self._calendar)
                and next_session is not None
            ):
                members = self._members_fn(t)
                universe_dfs: dict = {}
                for ticker in members:
                    if ticker not in self._data or ticker == spy_symbol:
                        continue
                    df = self._data[ticker]
                    sma_n = self._eng.stock_sma_filter
                    if sma_n > 0:
                        df_slice = df.loc[df.index <= t_ts]
                        if not compute_stock_above_sma(df_slice, sma_n):
                            continue
                    universe_dfs[ticker] = df

                target = select_momentum_stocks(universe_dfs, t_ts, top_n=self._eng.top_n)
                target_set = set(target)

                # Sell longs no longer in target
                for pos in book.long_positions():
                    if pos.ticker not in target_set:
                        book.add_pending(PendingOrder(pos.ticker, OrderSide.SELL, pos.shares, "momentum", "rebalance_exit", "long"))

                # Buy new long entries
                already_pending_long_buys = {
                    o.ticker for o in book.pending_orders()
                    if o.side == OrderSide.BUY and o.direction == "long"
                }
                for ticker in target:
                    if ticker in book.open_tickers:
                        continue
                    if ticker in already_pending_long_buys:
                        continue
                    if not risk_mgr.can_add_position(book, ticker):
                        continue

                    df_t = self._data.get(ticker)
                    if df_t is None:
                        continue
                    df_slice = df_t.loc[df_t.index <= t_ts]
                    if len(df_slice) < self._risk.min_history_days:
                        continue

                    vol = _stock_annual_vol(df_slice)
                    price = float(df_slice["adj_close"].iloc[-1])
                    shares = compute_shares_vol_target(
                        equity=mtm_equity,
                        price=price,
                        stock_annual_vol=vol,
                        target_portfolio_vol=self._eng.target_portfolio_vol,
                        n_positions=self._eng.top_n,
                        max_position_pct=self._risk.max_position_pct,
                        vol_floor=self._eng.vol_floor,
                    )
                    if shares == 0:
                        continue

                    min_pct = self._cost.min_position_commission_pct
                    if min_pct > 0:
                        trade_val_est = shares * price
                        comm_est = compute_commission(self._cost.commission_fixed, self._cost.commission_pct, trade_val_est)
                        if trade_val_est > 0 and comm_est / trade_val_est > min_pct:
                            continue

                    book.add_pending(PendingOrder(ticker, OrderSide.BUY, shares, "momentum", "rebalance_entry", "long"))

            # ── 9. Rebalance short book (always on rebalance day if short_n > 0) ──
            if (
                short_n > 0
                and not risk_mgr.is_stopped
                and is_rebalance_day(t, self._calendar)
                and next_session is not None
            ):
                members = self._members_fn(t)
                # Build universe without SMA filter for shorts (we want laggards)
                universe_dfs_short: dict = {}
                for ticker in members:
                    if ticker not in self._data or ticker == spy_symbol:
                        continue
                    universe_dfs_short[ticker] = self._data[ticker]

                short_target = select_short_stocks(universe_dfs_short, t_ts, short_n=short_n)
                short_target_set = set(short_target)

                # Cover shorts no longer in target
                for pos in book.short_positions():
                    if pos.ticker not in short_target_set:
                        book.add_pending(PendingOrder(pos.ticker, OrderSide.BUY, pos.shares, "momentum", "rebalance_exit", "short"))

                # Open new short entries
                already_pending_short_entries = {
                    o.ticker for o in book.pending_orders()
                    if o.side == OrderSide.SELL and o.direction == "short"
                }
                for ticker in short_target:
                    if ticker in book.short_tickers:
                        continue
                    if ticker in already_pending_short_entries:
                        continue
                    if not risk_mgr.can_add_position(book, ticker):
                        continue

                    df_t = self._data.get(ticker)
                    if df_t is None:
                        continue
                    df_slice = df_t.loc[df_t.index <= t_ts]
                    if len(df_slice) < self._risk.min_history_days:
                        continue

                    vol = _stock_annual_vol(df_slice)
                    price = float(df_slice["adj_close"].iloc[-1])
                    shares = compute_shares_vol_target(
                        equity=mtm_equity,
                        price=price,
                        stock_annual_vol=vol,
                        target_portfolio_vol=self._eng.target_portfolio_vol,
                        n_positions=short_n,
                        max_position_pct=self._risk.max_position_pct,
                        vol_floor=self._eng.vol_floor,
                    )
                    if shares == 0:
                        continue

                    min_pct = self._cost.min_position_commission_pct
                    if min_pct > 0:
                        trade_val_est = shares * price
                        comm_est = compute_commission(self._cost.commission_fixed, self._cost.commission_pct, trade_val_est)
                        if trade_val_est > 0 and comm_est / trade_val_est > min_pct:
                            continue

                    book.add_pending(PendingOrder(ticker, OrderSide.SELL, shares, "momentum", "rebalance_entry", "short"))

            # ── 10. Year-end tax ──────────────────────────────────────────────
            if i + 1 < len(sessions):
                if sessions[i + 1].year != t.year:
                    tax_usd = tax_ledger.year_end()
                    cash -= tax_usd
                    total_tax += tax_usd
            else:
                tax_usd = tax_ledger.year_end()
                cash -= tax_usd
                total_tax += tax_usd

            # ── 11. Log ───────────────────────────────────────────────────────
            equity_rows.append((t, mtm_equity))

        equity_df = pd.DataFrame(equity_rows, columns=["date", "equity"])
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.set_index("date")

        return RunResult(
            equity_curve=equity_df,
            trades=trades,
            cash=cash,
            total_commission_usd=total_comm,
            total_tax_usd=total_tax,
            total_slippage_usd=total_slip,
            n_trades=len(trades),
            start_date=start_date,
            end_date=end_date,
            engine="momentum",
        )
