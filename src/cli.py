"""
CLI entry point: python -m src.cli <command> <subcommand> [options]

Commands
--------
data update   [--start YYYY-MM-DD] [--end YYYY-MM-DD]
data report
data crosscheck
universe members --date YYYY-MM-DD
universe fetch-constituents
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_everything(config_dir: Path = Path("config")):
    """Load config + providers + cache + universe.  Lazy import to keep startup fast."""
    from .config import load_config
    from .data.cache import CacheManager
    from .data.calendar import TradingCalendar
    from .data.yfinance_provider import YFinanceProvider
    from .data.stooq_provider import StooqProvider
    from .universe.sp500 import SP500Universe
    from .universe.etfs import ETFUniverse

    cfg = load_config(config_dir)

    provider = YFinanceProvider()
    secondary = StooqProvider()
    cache = CacheManager(
        cfg.data.cache_dir,
        overlap_sessions=cfg.data.overlap_sessions,
        no_data_retry_days=cfg.data.no_data_retry_days,
        adj_tolerance=cfg.data.adj_tolerance,
        config_start_date=cfg.data.start_date,
    )
    etfs = ETFUniverse.from_config(cfg.universe.etfs)

    return cfg, provider, secondary, cache, etfs


def _build_calendar(cache, provider_name: str):
    from .data.calendar import TradingCalendar
    spy_df = cache.load(provider_name, "SPY")
    if spy_df is not None and not spy_df.empty:
        return TradingCalendar.from_spy_data(spy_df)
    return None


def cmd_data_update(args) -> int:
    from .config import load_config
    from .data.cache import CacheManager
    from .data.calendar import TradingCalendar
    from .data.integrity import run_all_checks
    from .data.yfinance_provider import YFinanceProvider
    from .universe.sp500 import SP500Universe
    from .universe.etfs import ETFUniverse
    from .universe.symbols import resolve_symbol

    cfg, provider, _, cache, etfs = _load_everything(Path(args.config_dir))

    # Override date range if provided
    start = date.fromisoformat(args.start) if args.start else cfg.data.start_date
    end = date.fromisoformat(args.end) if args.end else cfg.data.end_date

    # ── 1. Bootstrap calendar from SPY ──────────────────────────────────────
    logger.info("Fetching SPY to bootstrap trading calendar...")
    result = cache.update_symbol(provider, "SPY")
    logger.info("SPY: %s", result)

    calendar = _build_calendar(cache, provider.name)
    if calendar:
        cache.set_calendar(calendar)
        logger.info("Trading calendar: %d sessions", len(calendar))
    else:
        logger.warning("Could not build calendar from SPY; using date-based fallback")

    # ── 2. Load S&P 500 universe ─────────────────────────────────────────────
    constituent_path = Path(cfg.universe.constituent_source)
    if not constituent_path.exists():
        logger.error(
            "Constituent file not found: %s\n"
            "Run: python -m src.cli universe fetch-constituents",
            constituent_path,
        )
        return 1

    sp500 = SP500Universe.from_csv(constituent_path)
    ever = sp500.ever_members(start, end)
    logger.info("S&P 500 universe: %d ever-members from %s to %s", len(ever), start, end)

    # ── 3. Update all symbols ────────────────────────────────────────────────
    all_symbols = sorted(ever) + [
        s for s in etfs.symbols() if s not in ever and s != "SPY"
    ]

    run_id = cache.start_run(provider.name)
    counts = dict(attempted=0, updated=0, refetched=0, no_data=0, error=0, skipped=0)

    for i, ticker in enumerate(all_symbols, 1):
        resolved = resolve_symbol(ticker, cfg.universe.aliases)
        counts["attempted"] += 1
        try:
            outcome = cache.update_symbol(provider, resolved)
        except Exception as exc:
            logger.error("Unexpected error for %s: %s", resolved, exc)
            outcome = "error"

        counts[outcome] = counts.get(outcome, 0) + 1

        if outcome not in ("skipped",) or i % 100 == 0:
            logger.info(
                "[%d/%d] %s → %s", i, len(all_symbols), resolved, outcome
            )

        # Throttle
        if outcome not in ("skipped",):
            time.sleep(cfg.data.throttle_sleep)

    # ── 4. Run integrity checks ──────────────────────────────────────────────
    logger.info("Running integrity checks...")
    calendar_sessions: set[date] = set()
    if calendar:
        calendar_sessions = set(calendar.sessions(start, end))

    int_count = 0
    for ticker in all_symbols:
        resolved = resolve_symbol(ticker, cfg.universe.aliases)
        df = cache.load(provider.name, resolved)
        if df is None or df.empty:
            continue
        failures = run_all_checks(
            resolved, df, calendar_sessions,
            large_return_threshold=cfg.data.large_return_threshold,
            stale_price_run=cfg.data.stale_price_run,
        )
        for f in failures:
            cache.log_integrity_failure(
                run_id, resolved, f.date, f.check, f.detail
            )
        int_count += len(failures)

    cache.finish_run(
        run_id,
        symbols_attempted=counts["attempted"],
        symbols_updated=counts["updated"] + counts.get("refetched", 0),
        symbols_no_data=counts["no_data"],
        symbols_error=counts["error"],
        symbols_skipped=counts["skipped"],
    )

    logger.info(
        "Done. updated=%d  no_data=%d  error=%d  skipped=%d  integrity_failures=%d",
        counts["updated"] + counts.get("refetched", 0),
        counts["no_data"], counts["error"], counts["skipped"], int_count,
    )
    return 0


def cmd_data_report(args) -> int:
    from .data.calendar import TradingCalendar
    from .reports.data_report import generate_data_report
    from .universe.sp500 import SP500Universe
    from .universe.symbols import resolve_symbol

    cfg, _, _, cache, etfs = _load_everything(Path(args.config_dir))

    constituent_path = Path(cfg.universe.constituent_source)
    if not constituent_path.exists():
        logger.error("Constituent file not found: %s", constituent_path)
        return 1

    sp500 = SP500Universe.from_csv(constituent_path)
    calendar = _build_calendar(cache, cfg.data.provider)
    calendar_sessions = (
        set(calendar.sessions(cfg.data.start_date, cfg.data.end_date))
        if calendar else None
    )

    output = Path("reports") / "data_report.md"
    generate_data_report(
        cache=cache,
        sp500=sp500,
        etfs=etfs,
        provider_name=cfg.data.provider,
        config_start=cfg.data.start_date,
        config_end=cfg.data.end_date,
        constituent_sha256=cfg.universe.constituent_sha256,
        with_data_threshold=cfg.data.with_data_threshold,
        large_return_threshold=cfg.data.large_return_threshold,
        stale_price_run=cfg.data.stale_price_run,
        output_path=output,
        calendar_sessions=calendar_sessions,
    )
    print(f"Report written to {output}")
    return 0


def cmd_data_crosscheck(args) -> int:
    import json
    from .data.crosscheck import crosscheck_sample
    from .data.stooq_provider import StooqProvider
    from .data.yfinance_provider import YFinanceProvider
    from .universe.sp500 import SP500Universe
    from .universe.symbols import resolve_symbol

    cfg, provider, secondary, cache, etfs = _load_everything(Path(args.config_dir))

    constituent_path = Path(cfg.universe.constituent_source)
    sp500 = SP500Universe.from_csv(constituent_path) if constituent_path.exists() else None

    if sp500:
        symbols = sorted(
            resolve_symbol(t, cfg.universe.aliases)
            for t in sp500.ever_members(cfg.data.start_date, cfg.data.end_date)
        )
    else:
        symbols = etfs.symbols()

    result = crosscheck_sample(
        provider, secondary, symbols,
        sample_n=getattr(args, "sample_n", 20),
        seed=42,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_universe_members(args) -> int:
    from .universe.sp500 import SP500Universe

    cfg, _, _, _, _ = _load_everything(Path(args.config_dir))
    d = date.fromisoformat(args.date)

    constituent_path = Path(cfg.universe.constituent_source)
    if not constituent_path.exists():
        logger.error("Constituent file not found: %s", constituent_path)
        return 1

    sp500 = SP500Universe.from_csv(constituent_path)
    try:
        members = sorted(sp500.members_on(d))
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    print(f"S&P 500 members on {d} ({len(members)} tickers):")
    for i, t in enumerate(members, 1):
        end = "\n" if i % 10 == 0 else "  "
        print(t, end=end)
    print()
    return 0


def cmd_universe_fetch(args) -> int:
    from .universe.sp500 import fetch_constituents
    import yaml

    cfg, _, _, _, _ = _load_everything(Path(args.config_dir))
    dest = Path(cfg.universe.constituent_source)

    sha256, filename = fetch_constituents(dest)
    print(f"Downloaded: {filename}")
    print(f"SHA-256:    {sha256}")
    print(f"Saved to:   {dest}")

    # Update universe.yaml
    universe_yaml = Path(args.config_dir) / "universe.yaml"
    with open(universe_yaml) as f:
        text = f.read()
    text = text.replace(
        f'constituent_sha256: ""', f'constituent_sha256: "{sha256}"'
    ).replace(
        f"constituent_sha256: ''", f'constituent_sha256: "{sha256}"'
    )
    with open(universe_yaml, "w") as f:
        f.write(text)
    print(f"Updated {universe_yaml} with SHA-256.")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Short-horizon equity signal system — data layer CLI",
    )
    parser.add_argument(
        "--config-dir", default="config", metavar="DIR",
        help="Directory with data.yaml and universe.yaml (default: config)"
    )

    sub = parser.add_subparsers(dest="group", required=True)

    # ── data ──────────────────────────────────────────────────────────────────
    data_p = sub.add_parser("data", help="Data management commands")
    data_sub = data_p.add_subparsers(dest="subcommand", required=True)

    upd = data_sub.add_parser("update", help="Download/update daily OHLCV for the full universe")
    upd.add_argument("--start", default=None, metavar="YYYY-MM-DD")
    upd.add_argument("--end", default=None, metavar="YYYY-MM-DD")
    upd.set_defaults(func=cmd_data_update)

    rep = data_sub.add_parser("report", help="Generate Markdown data quality report")
    rep.set_defaults(func=cmd_data_report)

    cc = data_sub.add_parser("crosscheck", help="Cross-check yfinance vs Stooq on a sample")
    cc.add_argument("--sample-n", type=int, default=20)
    cc.set_defaults(func=cmd_data_crosscheck)

    # ── universe ───────────────────────────────────────────────────────────────
    uni_p = sub.add_parser("universe", help="Universe commands")
    uni_sub = uni_p.add_subparsers(dest="subcommand", required=True)

    mem = uni_sub.add_parser("members", help="List S&P 500 members on a given date")
    mem.add_argument("--date", required=True, metavar="YYYY-MM-DD")
    mem.set_defaults(func=cmd_universe_members)

    fetch = uni_sub.add_parser(
        "fetch-constituents",
        help="Download S&P 500 constituent CSV from fja05680/sp500",
    )
    fetch.set_defaults(func=cmd_universe_fetch)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
