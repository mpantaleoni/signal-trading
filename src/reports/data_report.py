"""
Generate the Markdown data-quality report (reports/data_report.md).

The report contains data-quality information only.
No returns or performance figures are included.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from ..data.cache import CacheManager
from ..data.integrity import IntegrityFailure, run_all_checks
from ..universe.etfs import ETFUniverse
from ..universe.sp500 import SP500Universe
from ..universe.symbols import resolve_symbol

logger = logging.getLogger(__name__)

_KNOWN_BIAS = (
    "KNOWN BIAS: free data under-represents delisted losers. "
    "All stock-universe results are an optimistic upper bound. "
    "This must be stated on every report."
)


def generate_data_report(
    cache: CacheManager,
    sp500: SP500Universe,
    etfs: ETFUniverse,
    provider_name: str,
    config_start: date,
    config_end: date,
    constituent_sha256: str,
    with_data_threshold: float,
    large_return_threshold: float,
    stale_price_run: int,
    output_path: Path,
    *,
    calendar_sessions: Optional[set[date]] = None,
    aliases: Optional[dict[str, str]] = None,
) -> str:
    """
    Build the Markdown report and write it to *output_path*.
    Returns the report text.
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# Data Quality Report",
        "",
        f"Run date: {date.today().isoformat()}",
        f"Provider: {provider_name}",
        f"Constituent-list SHA-256: `{constituent_sha256 or '(not set)'}`",
        f"Development period: {config_start.isoformat()} to {config_end.isoformat()}",
        "",
        f"> **{_KNOWN_BIAS}**",
        "",
    ]

    _aliases = aliases or {}

    # ── Per-year table ────────────────────────────────────────────────────────
    lines += _per_year_section(
        cache, sp500, provider_name, config_start, config_end,
        calendar_sessions, with_data_threshold,
        large_return_threshold, stale_price_run,
        _aliases,
    )

    # ── Missing members ───────────────────────────────────────────────────────
    lines += _missing_members_section(sp500, cache, provider_name, config_start, config_end, _aliases)

    # ── Top symbols by failure count ──────────────────────────────────────────
    lines += _top_failures_section(cache)

    # ── ETF section ───────────────────────────────────────────────────────────
    lines += _etf_section(cache, etfs, provider_name, calendar_sessions,
                          large_return_threshold, stale_price_run)

    text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    logger.info("Data report written to %s", output_path)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Section builders
# ──────────────────────────────────────────────────────────────────────────────

def _per_year_section(
    cache: CacheManager,
    sp500: SP500Universe,
    provider_name: str,
    start: date,
    end: date,
    calendar_sessions: Optional[set[date]],
    with_data_threshold: float,
    large_return_threshold: float,
    stale_price_run: int,
    aliases: dict[str, str],
) -> list[str]:
    lines = [
        "## Per-year coverage",
        "",
        "| Year | Members | With data | Partial | Missing | Missing % | Integrity failures |",
        "|------|---------|-----------|---------|---------|-----------|-------------------|",
    ]

    first_year = max(start.year, sp500.first_snapshot_date().year)
    last_year = end.year

    for year in range(first_year, last_year + 1):
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)

        try:
            ever = sp500.ever_members(year_start, year_end)
        except ValueError:
            continue

        intervals = sp500.membership_intervals()

        if calendar_sessions:
            year_sessions = {d for d in calendar_sessions if year_start <= d <= year_end}
        else:
            year_sessions = set()

        with_data = partial = missing_count = int_failures = 0

        for ticker in ever:
            resolved = resolve_symbol(ticker, aliases)
            df = cache.load(provider_name, resolved)

            # Compute in-index sessions for this ticker during this year
            if calendar_sessions and ticker in intervals:
                in_index = _in_index_sessions(
                    ticker, intervals[ticker], year_sessions
                )
            else:
                in_index = year_sessions

            if not in_index:
                # Can't assess without calendar; count as "no info"
                if df is None or df.empty:
                    missing_count += 1
                else:
                    with_data += 1
                continue

            if df is None or df.empty:
                actual_in_year = set()
            else:
                actual_in_year = {
                    (ts.date() if hasattr(ts, "date") else ts)
                    for ts in df.index
                    if year_start <= (ts.date() if hasattr(ts, "date") else ts) <= year_end
                }

            coverage = len(actual_in_year & in_index) / len(in_index) if in_index else 0.0

            if coverage >= with_data_threshold:
                with_data += 1
            elif coverage > 0:
                partial += 1
            else:
                missing_count += 1

            # Count integrity failures for this ticker in this year
            if df is not None and not df.empty:
                year_df = df[
                    df.index.map(lambda ts: year_start <= (ts.date() if hasattr(ts, "date") else ts) <= year_end)
                ]
                if not year_df.empty:
                    failures = run_all_checks(
                        ticker, year_df,
                        year_sessions,
                        large_return_threshold=large_return_threshold,
                        stale_price_run=stale_price_run,
                    )
                    int_failures += len(failures)

        distinct = len(ever)
        missing_pct = f"{missing_count / distinct:.1%}" if distinct else "n/a"
        lines.append(
            f"| {year} | {distinct} | {with_data} | {partial} | "
            f"{missing_count} | {missing_pct} | {int_failures} |"
        )

    lines.append("")
    return lines


def _missing_members_section(
    sp500: SP500Universe,
    cache: CacheManager,
    provider_name: str,
    start: date,
    end: date,
    aliases: dict[str, str],
) -> list[str]:
    lines = [
        "## Missing members (no price data)",
        "",
        "Tickers that were S&P 500 members during the data window but have "
        "no cached data.",
        "",
        "| Ticker | First membership | Last membership (or current) |",
        "|--------|-----------------|------------------------------|",
    ]

    intervals = sp500.membership_intervals()
    ever = sp500.ever_members(start, end)

    missing = []
    for ticker in sorted(ever):
        resolved = resolve_symbol(ticker, aliases)
        df = cache.load(provider_name, resolved)
        if df is None or df.empty:
            spans = intervals.get(ticker, [])
            first_start = spans[0][0].isoformat() if spans else "unknown"
            last_end = (
                spans[-1][1].isoformat() if spans and spans[-1][1] else "current"
            )
            missing.append((ticker, first_start, last_end))

    for ticker, fs, le in missing:
        lines.append(f"| {ticker} | {fs} | {le} |")

    lines += ["", f"Total missing: {len(missing)}", ""]
    return lines


def _top_failures_section(cache: CacheManager) -> list[str]:
    run_id = cache.latest_run_id()
    lines = [
        "## Top 20 symbols by integrity failure count",
        f"*(latest run: {run_id or 'n/a'})*",
        "",
        "| Rank | Ticker | Failures |",
        "|------|--------|---------|",
    ]

    all_failures = cache.all_integrity_failures(run_id=run_id)
    counts: dict[str, int] = defaultdict(int)
    for f in all_failures:
        counts[f["symbol"]] += 1

    top20 = sorted(counts.items(), key=lambda x: -x[1])[:20]
    for rank, (ticker, n) in enumerate(top20, 1):
        lines.append(f"| {rank} | {ticker} | {n} |")

    lines.append("")
    return lines


def _etf_section(
    cache: CacheManager,
    etfs: ETFUniverse,
    provider_name: str,
    calendar_sessions: Optional[set[date]],
    large_return_threshold: float,
    stale_price_run: int,
) -> list[str]:
    lines = [
        "## ETF coverage",
        "",
        "| ETF | Status | Rows | First date | Last date | Integrity failures |",
        "|-----|--------|------|-----------|----------|-------------------|",
    ]

    for etf in etfs.symbols():
        meta = cache.get_meta(provider_name, etf)
        df = cache.load(provider_name, etf)

        status = meta["status"] if meta else "not cached"
        rows = meta["rows"] if meta else 0
        first = meta["first_date"] if meta else ""
        last = meta["last_date"] if meta else ""

        if df is not None and not df.empty:
            sessions = calendar_sessions or set()
            failures = run_all_checks(
                etf, df, sessions,
                large_return_threshold=large_return_threshold,
                stale_price_run=stale_price_run,
            )
            n_fail = len(failures)
        else:
            n_fail = 0

        lines.append(
            f"| {etf} | {status} | {rows or ''} | {first} | {last} | {n_fail} |"
        )

    lines.append("")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _in_index_sessions(
    ticker: str,
    intervals: list[tuple[date, date | None]],
    year_sessions: set[date],
) -> set[date]:
    """Sessions in *year_sessions* that fall within *ticker*'s membership intervals."""
    result: set[date] = set()
    for start, end in intervals:
        for d in year_sessions:
            if start <= d and (end is None or d < end):
                result.add(d)
    return result
