"""Generate a Markdown backtest report from a RunResult."""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .metrics import (
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

# Bias warning — must appear on every stock-universe report (CLAUDE.md §3)
BIAS_WARNING = (
    "> **KNOWN BIAS**: free data under-represents delisted losers. "
    "All stock-universe results are an optimistic upper bound."
)


def _fmt_pct(v: float, decimals: int = 1) -> str:
    if math.isnan(v):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def _fmt_float(v: float, decimals: int = 2) -> str:
    if math.isnan(v) or math.isinf(v):
        return "—"
    return f"{v:.{decimals}f}"


def _fmt_usd(v: float) -> str:
    if math.isnan(v):
        return "—"
    return f"${v:,.0f}"


def generate_backtest_report(
    result,
    benchmark_result=None,
    n_trials: int = 1,
    output_path: Optional[Path] = None,
    title: str = "Backtest Report — Momentum Engine",
    stress_result=None,
) -> str:
    """Produce a Markdown report string from a RunResult.

    Parameters
    ----------
    result           : RunResult from BacktestEngine.run()
    benchmark_result : optional RunResult from run_spy_benchmark()
    n_trials         : number of variants tested (for DSR)
    output_path      : if given, write to this file
    title            : report title
    stress_result    : RunResult with doubled costs (for stress section)
    """
    eq = result.equity_curve["equity"]
    trades = result.trades

    s = sharpe(eq)
    so = sortino(eq)
    c = cagr(eq)
    tr = total_return(eq)
    mdd = max_drawdown(eq)
    mdd_dur = max_drawdown_duration(eq)
    n_obs = len(eq)
    dsr = deflated_sharpe_ratio(s if not math.isnan(s) else 0, n_trials, n_obs)

    # Benchmark metrics
    bench_sharpe = bench_cagr = bench_mdd = float("nan")
    if benchmark_result is not None:
        beq = benchmark_result.equity_curve["equity"]
        bench_sharpe = sharpe(beq)
        bench_cagr = cagr(beq)
        bench_mdd = max_drawdown(beq)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(BIAS_WARNING)
    lines.append("")
    lines.append(
        f"**Period**: {result.start_date} → {result.end_date}  "
        f"| **Initial capital**: $100,000"
    )
    lines.append("")

    # ── Summary metrics table ──────────────────────────────────────────────────
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Strategy | SPY Benchmark |")
    lines.append("|--------|----------|---------------|")
    lines.append(f"| Total Return | {_fmt_pct(tr)} | {_fmt_pct(bench_cagr)} |")
    lines.append(f"| CAGR | {_fmt_pct(c)} | {_fmt_pct(bench_cagr)} |")
    lines.append(f"| Sharpe Ratio | {_fmt_float(s)} | {_fmt_float(bench_sharpe)} |")
    lines.append(f"| Sortino Ratio | {_fmt_float(so)} | — |")
    lines.append(f"| Max Drawdown | {_fmt_pct(mdd)} | {_fmt_pct(bench_mdd)} |")
    lines.append(f"| Max DD Duration (days) | {mdd_dur} | — |")
    lines.append(f"| Deflated Sharpe (DSR) | {_fmt_float(dsr)} | — |")
    lines.append(f"| n_trials (DSR input) | {n_trials} | — |")
    lines.append("")

    # ── Cost and tax summary ───────────────────────────────────────────────────
    lines.append("## Costs & Tax")
    lines.append("")
    lines.append(f"- Total commission paid: {_fmt_usd(result.total_commission_usd)}")
    lines.append(f"- Total slippage: {_fmt_usd(result.total_slippage_usd)}")
    lines.append(f"- Total tax paid: {_fmt_usd(result.total_tax_usd)}")
    lines.append(f"- Total trades: {result.n_trades}")
    lines.append("")

    # ── Trade statistics ───────────────────────────────────────────────────────
    if trades:
        wr = win_rate(trades)
        aw = avg_win(trades)
        al = avg_loss(trades)
        ahd = avg_holding_days(trades)
        lines.append("## Trade Statistics")
        lines.append("")
        lines.append(f"- Number of trades: {len(trades)}")
        lines.append(f"- Win rate: {_fmt_pct(wr)}")
        lines.append(f"- Avg win: {_fmt_usd(aw)}")
        lines.append(f"- Avg loss: {_fmt_usd(al)}")
        lines.append(f"- Avg holding (sessions): {_fmt_float(ahd, 1)}")
        lines.append("")

    # ── Yearly returns table ───────────────────────────────────────────────────
    yr = yearly_returns(eq)
    bench_yr: dict[int, float] = {}
    if benchmark_result is not None:
        bench_yr = yearly_returns(benchmark_result.equity_curve["equity"])

    if yr:
        lines.append("## Yearly Returns")
        lines.append("")
        lines.append("| Year | Strategy | SPY Benchmark |")
        lines.append("|------|----------|---------------|")
        all_years = sorted(set(list(yr.keys()) + list(bench_yr.keys())))
        for y in all_years:
            strat = _fmt_pct(yr.get(y, float("nan")))
            bench = _fmt_pct(bench_yr.get(y, float("nan")))
            lines.append(f"| {y} | {strat} | {bench} |")
        lines.append("")

    # ── Stress test ────────────────────────────────────────────────────────────
    if stress_result is not None:
        seq = stress_result.equity_curve["equity"]
        lines.append("## Stress Test (2× Costs)")
        lines.append("")
        lines.append("| Metric | Stress |")
        lines.append("|--------|--------|")
        lines.append(f"| CAGR | {_fmt_pct(cagr(seq))} |")
        lines.append(f"| Sharpe | {_fmt_float(sharpe(seq))} |")
        lines.append(f"| Max Drawdown | {_fmt_pct(max_drawdown(seq))} |")
        lines.append("")

    # ── Equity curve (ASCII sparkline placeholder) ────────────────────────────
    lines.append("## Equity Curve")
    lines.append("")
    lines.append("*(Equity curve data in equity_curve attribute of RunResult)*")
    lines.append("")

    report = "\n".join(lines)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

    return report
