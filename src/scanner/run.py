"""Daily scanner — incremental live signal computation with persistent state.

Each run:
  1. Incremental price update (yfinance, today's members only)
  2. Fill yesterday's pending orders at today's open
  3. Mark-to-market equity + drawdown check
  4. Engine A exit scan  (if engine in {a, combined})
  5. Engine B rebalance  (if engine in {b, combined} and today is first session of month)
  6. Regime filter + entry scans
  7. Save state
  8. Generate status page
  9. Send Telegram alert

No look-ahead: signals use df.loc[:t_ts].
Fills use the next session's open (same guarantee as the backtester).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import load_config
from ..data.cache import CacheManager
from ..data.calendar import TradingCalendar
from ..portfolio.positions import OrderSide, PendingOrder, Position, PositionBook
from ..portfolio.risk import RiskManager, SectorMap
from ..portfolio.sizing import compute_shares
from ..signals.engine_a import compute_entry_signal, compute_exit_signals, compute_regime_ok
from ..signals.engine_b import is_rebalance_day, select_engine_b_stocks
from ..signals.indicators import atr
from ..backtest.fills import compute_commission, compute_fill_price
from ..backtest.tax import GermanTaxLedger
from .state import ScannerState

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("data/scanner_state.json")


@dataclass
class FillRecord:
    ticker: str
    side: str                     # "buy" | "sell"
    shares: int
    price: float
    engine: str
    reason: str
    pnl_usd: Optional[float] = None   # set for sells


@dataclass
class ScanReport:
    as_of_date: date
    equity: float
    cash: float
    initial_capital: float
    peak_equity: float
    drawdown_pct: float
    drawdown_status: str              # "ok" | "warning" | "stopped"
    fills_today: list[FillRecord] = field(default_factory=list)
    pending_orders: list[dict] = field(default_factory=list)
    open_positions: list[dict] = field(default_factory=list)
    trades_total: int = 0
    engine: str = "a"


class DailyScanner:
    """Live scanner — mirrors the backtester loop but persists state between runs.

    Parameters
    ----------
    config_dir : config/ directory (must contain data.yaml, universe.yaml, etc.)
    engine     : "a" | "b" | "combined"
    state_path : JSON file that persists PositionBook + cash between runs
    dry_run    : skip state save, status page write, and Telegram send
    """

    def __init__(
        self,
        config_dir: Path = Path("config"),
        engine: str = "a",
        state_path: Path = DEFAULT_STATE_PATH,
        dry_run: bool = False,
    ) -> None:
        self._config_dir = config_dir
        self._engine = engine.lower()
        self._state_path = state_path
        self._dry_run = dry_run

    # ── public entry point ────────────────────────────────────────────────────

    def scan(self) -> ScanReport:
        cfg = load_config(self._config_dir)
        from ..data.yfinance_provider import YFinanceProvider
        provider = YFinanceProvider()

        # ── price data update ─────────────────────────────────────────────────
        cache = CacheManager(
            cfg.data.cache_dir,
            overlap_sessions=cfg.data.overlap_sessions,
            no_data_retry_days=cfg.data.no_data_retry_days,
            adj_tolerance=cfg.data.adj_tolerance,
            config_start_date=cfg.data.start_date,
        )

        logger.info("Updating SPY...")
        cache.update_symbol(provider, "SPY")
        spy_df = cache.load(cfg.data.provider, "SPY")
        if spy_df is None or spy_df.empty:
            raise RuntimeError("SPY data unavailable")

        calendar = TradingCalendar.from_spy_data(spy_df)
        today = calendar.at_or_before(date.today())
        if today is None:
            raise RuntimeError("No trading sessions in calendar")

        # Update today's universe members only (not all-time ever)
        from ..universe.sp500 import SP500Universe
        from ..universe.symbols import resolve_symbol

        constituent_path = Path(cfg.universe.constituent_source)
        sp500 = SP500Universe.from_csv(constituent_path)

        try:
            members_today = {
                resolve_symbol(t, cfg.universe.aliases)
                for t in sp500.members_on(today)
            }
        except ValueError:
            members_today = set()

        for ticker in sorted(members_today):
            try:
                cache.update_symbol(provider, ticker)
            except Exception as exc:
                logger.warning("Skip %s: %s", ticker, exc)

        # Load price data into memory
        price_data: dict[str, pd.DataFrame] = {"SPY": spy_df}
        for ticker in members_today:
            df = cache.load(cfg.data.provider, ticker)
            if df is not None and not df.empty:
                price_data[ticker] = df

        logger.info("Loaded %d tickers", len(price_data))

        sectors_path = Path("data/raw/sp500_sectors.csv")
        sector_map = SectorMap.from_csv(sectors_path) if sectors_path.exists() else SectorMap.empty()

        def members_fn(d: date):
            try:
                return {resolve_symbol(t, cfg.universe.aliases) for t in sp500.members_on(d)}
            except ValueError:
                return set()

        # ── load state ────────────────────────────────────────────────────────
        state = ScannerState.load(self._state_path, cfg.risk.initial_capital)
        book = state.to_book()
        cash = state.cash

        tax_ledger = GermanTaxLedger(
            tax_rate=cfg.costs.tax_rate,
            annual_allowance_eur=cfg.costs.tax_annual_allowance,
        )
        risk_mgr = RiskManager(
            max_positions_engine_a=cfg.risk.max_positions_engine_a,
            max_positions_combined=cfg.risk.max_positions_combined,
            sector_cap=cfg.risk.sector_cap,
            drawdown_warning_pct=cfg.risk.drawdown_warning_pct,
            drawdown_stop_pct=cfg.risk.drawdown_stop_pct,
            initial_equity=state.peak_equity,
            sector_map=sector_map,
            max_positions_engine_b=cfg.risk.max_positions_engine_b,
        )

        # ── determine sessions to process ─────────────────────────────────────
        if state.as_of_date is None:
            sessions = [today]
        else:
            sessions = calendar.sessions(state.as_of_date + timedelta(days=1), today)

        if not sessions:
            logger.info("Already up to date (as_of=%s)", state.as_of_date)
            sessions = []

        spy_symbol = cfg.engine_a.spy_symbol
        fills_today: list[FillRecord] = []
        mtm_equity = cash + book.mark_to_market({})  # fallback if no sessions
        dd_status = "ok"

        # ── session loop ──────────────────────────────────────────────────────
        for t in sessions:
            t_ts = pd.Timestamp(t)
            next_session = calendar.next_session(t)

            # 1. Fill pending orders at t's open
            for order in book.flush_pending():
                if order.shares == 0:
                    continue
                df_tick = price_data.get(order.ticker)
                if df_tick is None:
                    continue
                row = df_tick.loc[df_tick.index <= t_ts]
                if row.empty or row.index[-1].date() != t:
                    continue
                raw_open = float(row["open"].iloc[-1])

                if order.side == OrderSide.BUY:
                    fill = compute_fill_price(raw_open, "buy", cfg.costs.slippage_bps)
                    trade_val = order.shares * fill
                    comm = compute_commission(cfg.costs.commission_fixed, cfg.costs.commission_pct, trade_val)
                    cash -= trade_val + comm
                    pos = Position(
                        ticker=order.ticker, shares=order.shares, entry_date=t,
                        entry_price=fill, entry_cost_usd=comm,
                        engine=order.engine, sector=sector_map.get(order.ticker),
                    )
                    try:
                        book.open_position(pos)
                        fills_today.append(FillRecord(order.ticker, "buy", order.shares, fill, order.engine, order.reason))
                    except ValueError:
                        cash += trade_val + comm  # already open — refund

                else:  # SELL
                    pos = book.get(order.ticker)
                    if pos is None:
                        continue
                    fill = compute_fill_price(raw_open, "sell", cfg.costs.slippage_bps)
                    trade_val = pos.shares * fill
                    comm = compute_commission(cfg.costs.commission_fixed, cfg.costs.commission_pct, trade_val)
                    realized_pnl = (fill - pos.entry_price) * pos.shares - pos.entry_cost_usd - comm
                    cash += trade_val - comm
                    tax_ledger.record_trade(realized_pnl)
                    fills_today.append(FillRecord(order.ticker, "sell", pos.shares, fill, pos.engine, order.reason, realized_pnl))
                    state.trades.append({
                        "ticker": order.ticker, "engine": pos.engine,
                        "entry_date": pos.entry_date.isoformat(), "exit_date": t.isoformat(),
                        "entry_price": pos.entry_price, "exit_price": fill,
                        "shares": pos.shares, "pnl_usd": realized_pnl, "exit_reason": order.reason,
                    })
                    book.close_position(order.ticker)

            # 2. MTM at close
            close_prices = _close_prices(book, price_data, t, t_ts)
            mtm_equity = cash + book.mark_to_market(close_prices)
            risk_mgr.update_equity(mtm_equity)
            dd_status = risk_mgr.check_drawdown(mtm_equity)

            # 3. Engine A exit scan
            if self._engine in ("a", "combined"):
                for pos in book.open_positions():
                    if pos.engine != "A":
                        continue
                    df_t = price_data.get(pos.ticker)
                    if df_t is None:
                        continue
                    df_slice = df_t.loc[df_t.index <= t_ts]
                    if df_slice.empty:
                        continue
                    exits = compute_exit_signals(df_slice, pos.entry_date, cfg.engine_a.max_hold_days, calendar)
                    reason = (
                        "sma5_exit" if exits["sma5_exit"] else
                        "sma200_exit" if exits["sma200_exit"] else
                        "time_stop" if exits["time_stop"] else None
                    )
                    if reason and next_session is not None:
                        book.add_pending(PendingOrder(pos.ticker, OrderSide.SELL, pos.shares, "A", reason))

            # 4. Regime filter
            spy_df_local = price_data.get(spy_symbol)
            regime_ok = False
            if spy_df_local is not None:
                regime_ok = compute_regime_ok(spy_df_local.loc[spy_df_local.index <= t_ts])

            # 5. Engine A entry scan
            if self._engine in ("a", "combined") and regime_ok and not risk_mgr.is_stopped and next_session:
                members = members_fn(t)
                pending_buy_tickers = {o.ticker for o in book.pending_orders() if o.side == OrderSide.BUY}
                for ticker in members:
                    if ticker in book.open_tickers or ticker in pending_buy_tickers or ticker == spy_symbol:
                        continue
                    if not risk_mgr.can_add_position(book, "A", ticker):
                        continue
                    df_t = price_data.get(ticker)
                    if df_t is None:
                        continue
                    df_slice = df_t.loc[df_t.index <= t_ts]
                    if len(df_slice) < cfg.risk.min_history_days:
                        continue
                    if not compute_entry_signal(df_slice, rsi_entry=cfg.engine_a.rsi_entry, min_adv_usd=0.0):
                        continue
                    shares = _size(df_slice, mtm_equity, cfg.risk)
                    if shares > 0:
                        book.add_pending(PendingOrder(ticker, OrderSide.BUY, shares, "A", "entry"))

            # 6. Engine B rebalance
            if self._engine in ("b", "combined") and is_rebalance_day(t, calendar) and next_session:
                if regime_ok and not risk_mgr.is_stopped:
                    members = members_fn(t)
                    universe_dfs = {tk: price_data[tk] for tk in members if tk in price_data and tk != spy_symbol}
                    target = select_engine_b_stocks(
                        universe_dfs, t_ts,
                        top_n=cfg.engine_b.top_n,
                        activity_fast=cfg.engine_b.activity_lookback_fast,
                        activity_slow=cfg.engine_b.activity_lookback_slow,
                        momentum_lookback=cfg.engine_b.momentum_lookback,
                        momentum_skip=cfg.engine_b.momentum_skip,
                        min_history=cfg.risk.min_history_days,
                    )
                    target_set = set(target)
                    for pos in book.open_positions():
                        if pos.engine == "B" and pos.ticker not in target_set:
                            book.add_pending(PendingOrder(pos.ticker, OrderSide.SELL, pos.shares, "B", "rebalance_exit"))
                    pending_buys = {o.ticker for o in book.pending_orders() if o.side == OrderSide.BUY}
                    for ticker in target:
                        if ticker in book.open_tickers or ticker in pending_buys:
                            continue
                        if not risk_mgr.can_add_position(book, "B", ticker):
                            continue
                        df_t = price_data.get(ticker)
                        if df_t is None:
                            continue
                        df_slice = df_t.loc[df_t.index <= t_ts]
                        if len(df_slice) < cfg.risk.min_history_days:
                            continue
                        shares = _size(df_slice, mtm_equity, cfg.risk)
                        if shares > 0:
                            book.add_pending(PendingOrder(ticker, OrderSide.BUY, shares, "B", "rebalance_entry"))
                else:
                    for pos in book.open_positions():
                        if pos.engine == "B":
                            book.add_pending(PendingOrder(pos.ticker, OrderSide.SELL, pos.shares, "B", "regime_exit"))

            # 7. Year-end tax
            nxt = calendar.next_session(t)
            if nxt is not None and nxt.year != t.year:
                cash -= tax_ledger.year_end()
            elif nxt is None:
                cash -= tax_ledger.year_end()

            state.append_equity(t, mtm_equity)

        # ── build report ──────────────────────────────────────────────────────
        state.as_of_date = today
        state.cash = cash
        state.peak_equity = max(state.peak_equity, mtm_equity)
        state.sync_from_book(book)

        final_close = _close_prices(book, price_data, today, pd.Timestamp(today))
        final_equity = cash + book.mark_to_market(final_close)
        dd_pct = max(0.0, 1.0 - final_equity / state.peak_equity) if state.peak_equity > 0 else 0.0
        cfg_risk = cfg.risk
        dd_final = (
            "stopped" if dd_pct >= cfg_risk.drawdown_stop_pct else
            "warning" if dd_pct >= cfg_risk.drawdown_warning_pct else
            "ok"
        )

        open_positions_report = []
        for pos in book.open_positions():
            cur = final_close.get(pos.ticker, pos.entry_price)
            unreal = (cur - pos.entry_price) * pos.shares
            pct = (cur / pos.entry_price - 1) * 100 if pos.entry_price > 0 else 0.0
            open_positions_report.append({
                "ticker": pos.ticker, "engine": pos.engine,
                "shares": pos.shares, "entry_date": pos.entry_date.isoformat(),
                "entry_price": pos.entry_price, "current_price": cur,
                "unrealized_pnl": unreal, "unrealized_pct": pct,
                "sessions_held": len(calendar.sessions(pos.entry_date, today)),
            })

        report = ScanReport(
            as_of_date=today,
            equity=final_equity,
            cash=cash,
            initial_capital=cfg.risk.initial_capital,
            peak_equity=state.peak_equity,
            drawdown_pct=dd_pct,
            drawdown_status=dd_final,
            fills_today=fills_today,
            pending_orders=[
                {"ticker": o.ticker, "side": o.side.value, "shares": o.shares,
                 "engine": o.engine, "reason": o.reason}
                for o in book.pending_orders()
            ],
            open_positions=open_positions_report,
            trades_total=len(state.trades),
            engine=self._engine,
        )

        if not self._dry_run:
            state.save(self._state_path)

        return report


# ── helpers ───────────────────────────────────────────────────────────────────


def _close_prices(book: PositionBook, price_data: dict, t: date, t_ts: pd.Timestamp) -> dict[str, float]:
    out: dict[str, float] = {}
    for ticker in book.open_tickers:
        df = price_data.get(ticker)
        if df is not None:
            row = df.loc[df.index <= t_ts]
            if not row.empty and row.index[-1].date() == t:
                out[ticker] = float(row["adj_close"].iloc[-1])
    return out


def _size(df_slice: pd.DataFrame, equity: float, risk_cfg) -> int:
    from ..signals.indicators import atr as _atr
    atr_s = _atr(df_slice["high"], df_slice["low"], df_slice["adj_close"], 20)
    atr_val = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
    price = float(df_slice["adj_close"].iloc[-1])
    return compute_shares(equity, price, atr_val, risk_cfg.risk_per_trade, risk_cfg.max_position_pct, risk_cfg.atr_floor)
