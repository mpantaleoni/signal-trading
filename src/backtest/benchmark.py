"""SPY buy-and-hold benchmark.

Buy at the open of the first session on or after start_date.
Sell at the close of the last session on or before end_date.
Apply entry commission + slippage on buy, exit commission + slippage on sell.
Tax only at final exit (not annually).

Returns a RunResult with equity_curve showing daily MTM.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from ..data.calendar import TradingCalendar
from ..portfolio.sizing import compute_shares
from .engine import RunResult
from .fills import EUR_USD_RATE, compute_commission, compute_fill_price
from .tax import GermanTaxLedger


def run_spy_benchmark(
    spy_df: pd.DataFrame,
    calendar: TradingCalendar,
    start_date: date,
    end_date: date,
    initial_capital: float,
    cost_cfg,
    stress: bool = False,
) -> RunResult:
    """Run a SPY buy-and-hold backtest over [start_date, end_date]."""
    sessions = calendar.sessions(start_date, end_date)
    if not sessions:
        raise ValueError(f"No sessions for benchmark in [{start_date}, {end_date}]")

    multiplier = cost_cfg.stress_cost_multiplier if stress else 1.0
    cash = float(initial_capital)
    shares_held = 0
    entry_price = 0.0
    entry_comm = 0.0
    total_comm = 0.0
    total_slip = 0.0

    def _adj_close_on(d: date) -> Optional[float]:
        ts = pd.Timestamp(d)
        row = spy_df.loc[spy_df.index <= ts]
        if row.empty or row.index[-1].date() != d:
            return None
        return float(row["adj_close"].iloc[-1])

    def _open_on(d: date) -> Optional[float]:
        ts = pd.Timestamp(d)
        row = spy_df.loc[spy_df.index == ts]
        if row.empty:
            return None
        return float(row["open"].iloc[0])

    # Buy at open of first session
    first_session = sessions[0]
    raw_open = _open_on(first_session)
    if raw_open is None or raw_open <= 0:
        raw_open = _adj_close_on(first_session) or 100.0

    buy_price = compute_fill_price(raw_open, "buy", cost_cfg.slippage_bps, multiplier)
    shares_held = int(cash / buy_price)  # whole shares
    if shares_held < 1:
        shares_held = 1
    trade_val = shares_held * buy_price
    entry_comm = compute_commission(cost_cfg.commission_fixed, cost_cfg.commission_pct, trade_val, multiplier)
    cash -= trade_val + entry_comm
    total_comm += entry_comm
    total_slip += shares_held * abs(buy_price - raw_open)
    entry_price = buy_price

    equity_rows: list[tuple[date, float]] = []

    for i, t in enumerate(sessions):
        adj = _adj_close_on(t)
        if adj is None:
            adj = entry_price  # fallback

        mtm = cash + shares_held * adj
        equity_rows.append((t, mtm))

    # Sell at close of last session
    last_session = sessions[-1]
    last_adj = _adj_close_on(last_session) or entry_price
    sell_price = compute_fill_price(last_adj, "sell", cost_cfg.slippage_bps, multiplier)
    exit_comm = compute_commission(cost_cfg.commission_fixed, cost_cfg.commission_pct, shares_held * sell_price, multiplier)
    total_comm += exit_comm
    total_slip += shares_held * abs(last_adj - sell_price)

    # Tax at final exit
    tax_ledger = GermanTaxLedger(
        tax_rate=cost_cfg.tax_rate,
        annual_allowance_eur=cost_cfg.tax_annual_allowance,
    )
    realized_pnl = (sell_price - entry_price) * shares_held - entry_comm - exit_comm
    tax_ledger.record_trade(realized_pnl)
    tax_usd = tax_ledger.year_end()

    final_cash = cash + shares_held * sell_price - exit_comm - tax_usd

    equity_df = pd.DataFrame(equity_rows, columns=["date", "equity"])
    equity_df["date"] = pd.to_datetime(equity_df["date"])
    equity_df = equity_df.set_index("date")
    # Update last equity value to reflect sale
    if not equity_df.empty:
        equity_df.iloc[-1] = final_cash

    return RunResult(
        equity_curve=equity_df,
        trades=[{
            "ticker": "SPY",
            "engine": "benchmark",
            "entry_date": first_session,
            "exit_date": last_session,
            "entry_price": entry_price,
            "exit_price": sell_price,
            "shares": shares_held,
            "pnl_usd": realized_pnl,
            "exit_reason": "end_of_period",
            "sessions_held": len(sessions),
        }],
        cash=final_cash,
        total_commission_usd=total_comm,
        total_tax_usd=tax_usd,
        total_slippage_usd=total_slip,
        n_trades=1,
        start_date=start_date,
        end_date=end_date,
        engine="SPY_benchmark",
    )
