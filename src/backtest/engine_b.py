"""Engine B event-driven monthly-rebalance backtester.

Event loop (each session t):
  1. Process fills  — pending orders → open at t's open price
  2. Mark-to-market — update peak equity, check drawdown
  3. If rebalance day:
       a. Compute regime (SPY close vs SMA200)
       b. If regime OK and not stopped → select new holdings → queue sells + buys
       c. If regime bad or stopped    → queue sells for all B positions
  4. Year-end tax
  5. Log

No mid-month exits: positions are held until the next rebalance.
No look-ahead: all signals use df.loc[:t_ts].
Fills use the next session's open price (t+1).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

from ..data.calendar import TradingCalendar
from ..portfolio.positions import OrderSide, PendingOrder, Position, PositionBook
from ..portfolio.risk import RiskManager, SectorMap
from ..portfolio.sizing import compute_shares
from ..signals.engine_a import compute_regime_ok
from ..signals.engine_b import is_rebalance_day, select_engine_b_stocks
from ..signals.indicators import atr
from .engine import RunResult
from .fills import compute_commission, compute_fill_price
from .tax import GermanTaxLedger

logger = logging.getLogger(__name__)


class EngineBBacktest:
    """Monthly-rebalance backtester for Engine B.

    Parameters
    ----------
    price_data      : dict[ticker, DataFrame] — canonical OHLCV, indexed by tz-naive Timestamp
    calendar        : TradingCalendar
    universe_members: callable(date) -> set[str] — returns members on a given date
    initial_capital : starting cash in USD
    engine_cfg      : EngineBConfig
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
            max_positions_engine_a=self._risk.max_positions_engine_a,
            max_positions_combined=self._risk.max_positions_combined,
            sector_cap=self._risk.sector_cap,
            drawdown_warning_pct=self._risk.drawdown_warning_pct,
            drawdown_stop_pct=self._risk.drawdown_stop_pct,
            initial_equity=cash,
            sector_map=self._sector_map,
            max_positions_engine_b=self._risk.max_positions_engine_b,
        )

        equity_rows: list[tuple[date, float]] = []
        trades: list[dict] = []
        total_comm = 0.0
        total_tax = 0.0
        total_slip = 0.0

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

                if order.side == OrderSide.BUY:
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
                    )
                    try:
                        book.open_position(pos)
                    except ValueError:
                        cash += trade_val + comm
                        total_comm -= comm
                        total_slip -= abs(slip_cost)

                else:  # SELL
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

            # ── 2. Mark-to-market equity ──────────────────────────────────────
            close_prices: dict[str, float] = {}
            for ticker in book.open_tickers:
                df_t = self._data.get(ticker)
                if df_t is not None:
                    row = df_t.loc[df_t.index <= t_ts]
                    if not row.empty and row.index[-1].date() == t:
                        close_prices[ticker] = float(row["adj_close"].iloc[-1])

            mtm_positions = book.mark_to_market(close_prices)
            mtm_equity = cash + mtm_positions
            risk_mgr.update_equity(mtm_equity)
            dd_status = risk_mgr.check_drawdown(mtm_equity)
            if dd_status == "warning":
                logger.debug("%s drawdown warning: %.1f%%", t, risk_mgr.drawdown_from_peak(mtm_equity) * 100)

            # ── 3. Rebalance if first session of month ───────────────────────
            if is_rebalance_day(t, self._calendar) and next_session is not None:
                spy_symbol = self._eng.spy_symbol
                spy_df = self._data.get(spy_symbol)
                regime_ok = False
                if spy_df is not None:
                    spy_slice = spy_df.loc[spy_df.index <= t_ts]
                    regime_ok = compute_regime_ok(spy_slice)

                if regime_ok and not risk_mgr.is_stopped:
                    # Compute new target holdings
                    members = self._members_fn(t)
                    universe_dfs = {
                        ticker: self._data[ticker]
                        for ticker in members
                        if ticker in self._data and ticker != spy_symbol
                    }
                    target = select_engine_b_stocks(
                        universe_dfs,
                        t_ts,
                        top_n=self._eng.top_n,
                        activity_fast=self._eng.activity_lookback_fast,
                        activity_slow=self._eng.activity_lookback_slow,
                        momentum_lookback=self._eng.momentum_lookback,
                        momentum_skip=self._eng.momentum_skip,
                        min_history=self._risk.min_history_days,
                    )
                    target_set = set(target)

                    # Queue sells for positions not in new target
                    for pos in book.open_positions():
                        if pos.engine != "B":
                            continue
                        if pos.ticker not in target_set:
                            book.add_pending(PendingOrder(pos.ticker, OrderSide.SELL, pos.shares, "B", "rebalance_exit"))

                    # Queue buys for new entries
                    already_pending_buys = {
                        o.ticker for o in book.pending_orders()
                        if o.side == OrderSide.BUY
                    }
                    for ticker in target:
                        if ticker in book.open_tickers:
                            continue
                        if ticker in already_pending_buys:
                            continue
                        if not risk_mgr.can_add_position(book, "B", ticker):
                            continue

                        df_t = self._data.get(ticker)
                        if df_t is None:
                            continue
                        df_slice = df_t.loc[df_t.index <= t_ts]
                        if len(df_slice) < self._risk.min_history_days:
                            continue

                        atr_series = atr(df_slice["high"], df_slice["low"], df_slice["adj_close"], 20)
                        atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
                        price = float(df_slice["adj_close"].iloc[-1])

                        shares = compute_shares(
                            equity=mtm_equity,
                            price=price,
                            atr_value=atr_val,
                            risk_per_trade=self._risk.risk_per_trade,
                            max_position_pct=self._risk.max_position_pct,
                            atr_floor=self._risk.atr_floor,
                        )
                        if shares == 0:
                            continue

                        book.add_pending(PendingOrder(ticker, OrderSide.BUY, shares, "B", "rebalance_entry"))

                else:
                    # Regime bad or stopped — exit all B positions
                    for pos in book.open_positions():
                        if pos.engine == "B":
                            book.add_pending(PendingOrder(pos.ticker, OrderSide.SELL, pos.shares, "B", "regime_exit"))

            # ── 4. Year-end tax ───────────────────────────────────────────────
            if i + 1 < len(sessions):
                if sessions[i + 1].year != t.year:
                    tax_usd = tax_ledger.year_end()
                    cash -= tax_usd
                    total_tax += tax_usd
            else:
                tax_usd = tax_ledger.year_end()
                cash -= tax_usd
                total_tax += tax_usd

            # ── 5. Log ────────────────────────────────────────────────────────
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
            engine="B",
        )
