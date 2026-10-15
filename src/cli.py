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

    membership_intervals = sp500.membership_intervals()

    int_count = 0
    for ticker in all_symbols:
        resolved = resolve_symbol(ticker, cfg.universe.aliases)
        df = cache.load(provider.name, resolved)
        if df is None or df.empty:
            continue

        # Bound checks to S&P 500 membership periods to avoid noise from
        # non-US trading days in pre-NYSE data (e.g. FERG, AMCR).
        if ticker in membership_intervals:
            _intervals = membership_intervals[ticker]
            member_sessions = {
                d for d in calendar_sessions
                for (s, e) in _intervals
                if s <= d and (e is None or d < e)
            } if calendar_sessions else set()
            df_check = df[df.index.map(
                lambda ts: any(
                    s <= (ts.date() if hasattr(ts, "date") else ts)
                    and (e is None or (ts.date() if hasattr(ts, "date") else ts) < e)
                    for (s, e) in _intervals
                )
            )]
        else:
            member_sessions = calendar_sessions
            df_check = df

        if df_check.empty:
            continue

        failures = run_all_checks(
            resolved, df_check, member_sessions,
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
        aliases=dict(cfg.universe.aliases or {}),
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


HOLDOUT_START = date(2019, 1, 1)
HOLDOUT_MSG = "Hold-out period starts 2019-01-01. Run refused. See CLAUDE.md §6."


def _load_backtest_deps(config_dir: Path):
    """Load everything needed for a backtest run."""
    from .config import load_config
    from .data.cache import CacheManager
    from .data.calendar import TradingCalendar
    from .portfolio.risk import SectorMap
    from .universe.sp500 import SP500Universe

    cfg = load_config(config_dir)
    cache = CacheManager(
        cfg.data.cache_dir,
        overlap_sessions=cfg.data.overlap_sessions,
        no_data_retry_days=cfg.data.no_data_retry_days,
        adj_tolerance=cfg.data.adj_tolerance,
        config_start_date=cfg.data.start_date,
    )

    # Build trading calendar from SPY
    spy_df = cache.load(cfg.data.provider, "SPY")
    if spy_df is None or spy_df.empty:
        logger.error("SPY data not found. Run: python -m src.cli data update")
        return None, None, None, None, None, None

    from .data.calendar import TradingCalendar
    calendar = TradingCalendar.from_spy_data(spy_df)

    # Load S&P 500 universe
    constituent_path = Path(cfg.universe.constituent_source)
    if not constituent_path.exists():
        logger.error("Constituent file not found: %s", constituent_path)
        return None, None, None, None, None, None

    from .universe.sp500 import SP500Universe
    sp500 = SP500Universe.from_csv(constituent_path)

    # Try to load sector map
    sectors_path = Path("data/raw/sp500_sectors.csv")
    if sectors_path.exists():
        sector_map = SectorMap.from_csv(sectors_path)
        logger.info("Loaded sector map from %s", sectors_path)
    else:
        from .portfolio.risk import SectorMap as SM
        sector_map = SM.empty()
        logger.warning("Sector map not found at %s; all tickers will use 'Unknown' sector", sectors_path)

    return cfg, cache, calendar, sp500, sector_map, spy_df


def _load_price_data(cache, cfg, sp500, start_date, end_date):
    """Load all cached OHLCV data into memory as dict[ticker, DataFrame]."""
    from .universe.symbols import resolve_symbol
    ever = sp500.ever_members(start_date, end_date)
    all_tickers = sorted(ever) + ["SPY"]
    price_data = {}
    for ticker in all_tickers:
        resolved = resolve_symbol(ticker, cfg.universe.aliases)
        df = cache.load(cfg.data.provider, resolved)
        if df is not None and not df.empty:
            price_data[resolved] = df
    logger.info("Loaded %d tickers into memory", len(price_data))
    return price_data


def cmd_backtest_run(args) -> int:
    from .backtest.benchmark import run_spy_benchmark
    from .backtest.engine import BacktestEngine
    from .backtest.registry import ExperimentRegistry
    from .reports.backtest_report import generate_backtest_report
    from .universe.symbols import resolve_symbol

    # Hold-out guard
    end_date = date.fromisoformat(args.end)
    if end_date >= HOLDOUT_START:
        logger.error(HOLDOUT_MSG)
        return 1

    start_date = date.fromisoformat(args.start)
    config_dir = Path(args.config_dir)
    engine_name = getattr(args, "engine", "a").lower()

    deps = _load_backtest_deps(config_dir)
    if deps[0] is None:
        return 1
    cfg, cache, calendar, sp500, sector_map, spy_df = deps

    price_data = _load_price_data(cache, cfg, sp500, start_date, end_date)

    def members_fn(d):
        from .universe.symbols import resolve_symbol
        try:
            return {
                resolve_symbol(t, cfg.universe.aliases)
                for t in sp500.members_on(d)
            }
        except ValueError:
            return set()

    if engine_name == "a":
        engine = BacktestEngine(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=cfg.risk.initial_capital,
            engine_cfg=cfg.engine_a,
            risk_cfg=cfg.risk,
            cost_cfg=cfg.costs,
            stress=getattr(args, "stress", False),
            sector_map=sector_map,
        )
        logger.info("Running backtest: Engine A | %s → %s", start_date, end_date)
        result = engine.run(start_date, end_date)

    elif engine_name == "b":
        from .backtest.engine_b import EngineBBacktest
        engine = EngineBBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=cfg.risk.initial_capital,
            engine_cfg=cfg.engine_b,
            risk_cfg=cfg.risk,
            cost_cfg=cfg.costs,
            stress=getattr(args, "stress", False),
            sector_map=sector_map,
        )
        logger.info("Running backtest: Engine B | %s → %s", start_date, end_date)
        result = engine.run(start_date, end_date)

    elif engine_name == "combined":
        from .backtest.combined_engine import CombinedBacktest
        engine = CombinedBacktest(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=cfg.risk.initial_capital,
            engine_a_cfg=cfg.engine_a,
            engine_b_cfg=cfg.engine_b,
            risk_cfg=cfg.risk,
            cost_cfg=cfg.costs,
            stress=getattr(args, "stress", False),
            sector_map=sector_map,
        )
        logger.info("Running backtest: Combined Engine | %s → %s", start_date, end_date)
        result = engine.run(start_date, end_date)

    else:
        logger.error("Unknown engine: %s", engine_name)
        return 1

    logger.info(
        "Done: %d trades | commission=$%.0f | tax=$%.0f",
        result.n_trades, result.total_commission_usd, result.total_tax_usd,
    )

    # Benchmark
    logger.info("Running SPY benchmark...")
    bench = run_spy_benchmark(spy_df, calendar, start_date, end_date, cfg.risk.initial_capital, cfg.costs)

    # Stress run (only if not already in stress mode)
    stress_result = None
    if not getattr(args, "stress", False):
        if engine_name == "a":
            stress_engine = BacktestEngine(
                price_data=price_data,
                calendar=calendar,
                universe_members=members_fn,
                initial_capital=cfg.risk.initial_capital,
                engine_cfg=cfg.engine_a,
                risk_cfg=cfg.risk,
                cost_cfg=cfg.costs,
                stress=True,
                sector_map=sector_map,
            )
            stress_result = stress_engine.run(start_date, end_date)
        elif engine_name == "b":
            from .backtest.engine_b import EngineBBacktest
            stress_engine = EngineBBacktest(
                price_data=price_data,
                calendar=calendar,
                universe_members=members_fn,
                initial_capital=cfg.risk.initial_capital,
                engine_cfg=cfg.engine_b,
                risk_cfg=cfg.risk,
                cost_cfg=cfg.costs,
                stress=True,
                sector_map=sector_map,
            )
            stress_result = stress_engine.run(start_date, end_date)
        elif engine_name == "combined":
            from .backtest.combined_engine import CombinedBacktest
            stress_engine = CombinedBacktest(
                price_data=price_data,
                calendar=calendar,
                universe_members=members_fn,
                initial_capital=cfg.risk.initial_capital,
                engine_a_cfg=cfg.engine_a,
                engine_b_cfg=cfg.engine_b,
                risk_cfg=cfg.risk,
                cost_cfg=cfg.costs,
                stress=True,
                sector_map=sector_map,
            )
            stress_result = stress_engine.run(start_date, end_date)

    # Registry
    registry = ExperimentRegistry(Path("data/experiments.db"))
    from .backtest.walk_forward import N_TRIALS
    exp_id = registry.log_run(
        engine=engine_name.upper() if engine_name != "combined" else "combined",
        start_date=start_date,
        end_date=end_date,
        params={"engine": engine_name},
        metrics={"n_trades": result.n_trades},
    )
    registry.close()

    # Generate report
    output_path = Path("reports") / f"backtest_engine_{engine_name}_{start_date}_{end_date}.md"
    report = generate_backtest_report(
        result=result,
        benchmark_result=bench,
        n_trials=N_TRIALS,
        output_path=output_path,
        stress_result=stress_result,
    )
    print(f"Report written to {output_path}")
    print(f"Trades: {result.n_trades}")
    return 0


def _print_and_write_wf_report(wf_result, engine_label: str, n_trials: int, output_path) -> None:
    import math
    print(f"\n=== Walk-Forward Summary — {engine_label} ===")
    print(f"{'Fold':>4} {'Train':>12} {'Test':>12} {'Train SR':>10} {'Test SR':>10} {'Test CAGR':>10} {'Max DD':>8} {'Best Params'}")
    print("-" * 100)
    for fold in wf_result.folds:
        train_sr = fold.train_sharpe if not math.isnan(fold.train_sharpe) else float("nan")
        test_sr = fold.test_sharpe if not math.isnan(fold.test_sharpe) else float("nan")
        test_cagr = fold.test_cagr if not math.isnan(fold.test_cagr) else float("nan")
        test_mdd = fold.test_max_dd if not math.isnan(fold.test_max_dd) else float("nan")
        print(
            f"{fold.fold_index:>4} "
            f"{str(fold.train_start)[:10]:>12} "
            f"{str(fold.test_start)[:10]:>12} "
            f"{train_sr:>10.3f} "
            f"{test_sr:>10.3f} "
            f"{test_cagr * 100:>9.1f}% "
            f"{test_mdd * 100:>7.1f}% "
            f"  {fold.best_params}"
        )
    print("-" * 100)
    print(f"Avg test Sharpe: {wf_result.avg_test_sharpe:.3f}")
    print(f"Avg test CAGR: {wf_result.avg_test_cagr * 100:.1f}%")
    print(f"n_trials (for DSR): {n_trials}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Walk-Forward Evaluation — {engine_label}",
        "",
        "> **KNOWN BIAS**: free data under-represents delisted losers. All results are an optimistic upper bound.",
        "",
        "## Configuration",
        "",
        f"- Folds: 9 (train 5 years, test 1 year, rolling by 1 year)",
        f"- n_trials (for DSR): {n_trials}",
        "",
        "## Fold Results",
        "",
        "| Fold | Train | Test | Train SR | Test SR | Test CAGR | Max DD | Best Params |",
        "|------|-------|------|----------|---------|-----------|--------|-------------|",
    ]
    for fold in wf_result.folds:
        lines.append(
            f"| {fold.fold_index} "
            f"| {fold.train_start}–{fold.train_end} "
            f"| {fold.test_start}–{fold.test_end} "
            f"| {fold.train_sharpe:.3f} "
            f"| {fold.test_sharpe:.3f} "
            f"| {fold.test_cagr * 100:.1f}% "
            f"| {fold.test_max_dd * 100:.1f}% "
            f"| {fold.best_params} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- Avg test Sharpe: {wf_result.avg_test_sharpe:.3f}",
        f"- Avg test CAGR: {wf_result.avg_test_cagr * 100:.1f}%",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {output_path}")


def cmd_backtest_walk_forward(args) -> int:
    from .backtest.registry import ExperimentRegistry
    from pathlib import Path

    engine_name = getattr(args, "engine", "a").lower()
    config_dir = Path(args.config_dir)
    deps = _load_backtest_deps(config_dir)
    if deps[0] is None:
        return 1
    cfg, cache, calendar, sp500, sector_map, spy_df = deps

    start_date = date(2005, 1, 1)
    end_date = date(2018, 12, 31)
    price_data = _load_price_data(cache, cfg, sp500, start_date, end_date)

    def members_fn(d):
        from .universe.symbols import resolve_symbol
        try:
            return {
                resolve_symbol(t, cfg.universe.aliases)
                for t in sp500.members_on(d)
            }
        except ValueError:
            return set()

    if engine_name == "a":
        from .backtest.walk_forward import WalkForwardEvaluator, N_TRIALS
        evaluator = WalkForwardEvaluator(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=cfg.risk.initial_capital,
            base_engine_cfg=cfg.engine_a,
            risk_cfg=cfg.risk,
            cost_cfg=cfg.costs,
            sector_map=sector_map,
        )
        logger.info("Starting walk-forward evaluation: Engine A (9 folds)...")
        wf_result = evaluator.run()
        n_trials = N_TRIALS
        engine_label = "Engine A"
        output_path = Path("reports") / "walk_forward_engine_a.md"

    elif engine_name == "b":
        from .backtest.walk_forward_b import WalkForwardEvaluatorB, N_TRIALS_B
        evaluator = WalkForwardEvaluatorB(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=cfg.risk.initial_capital,
            base_engine_cfg=cfg.engine_b,
            risk_cfg=cfg.risk,
            cost_cfg=cfg.costs,
            sector_map=sector_map,
        )
        logger.info("Starting walk-forward evaluation: Engine B (9 folds × 3 params = 27 trials)...")
        wf_result = evaluator.run()
        n_trials = N_TRIALS_B
        engine_label = "Engine B"
        output_path = Path("reports") / "walk_forward_engine_b.md"

    elif engine_name == "combined":
        from .backtest.walk_forward import WalkForwardEvaluator, N_TRIALS
        from .backtest.walk_forward_b import WalkForwardEvaluatorB, N_TRIALS_B
        from .backtest.walk_forward_combined import WalkForwardCombined, N_TRIALS_COMBINED

        logger.info("Running Engine A walk-forward for combined evaluation...")
        wf_a_evaluator = WalkForwardEvaluator(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=cfg.risk.initial_capital,
            base_engine_cfg=cfg.engine_a,
            risk_cfg=cfg.risk,
            cost_cfg=cfg.costs,
            sector_map=sector_map,
        )
        wf_a_result = wf_a_evaluator.run()

        logger.info("Running Engine B walk-forward for combined evaluation...")
        wf_b_evaluator = WalkForwardEvaluatorB(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=cfg.risk.initial_capital,
            base_engine_cfg=cfg.engine_b,
            risk_cfg=cfg.risk,
            cost_cfg=cfg.costs,
            sector_map=sector_map,
        )
        wf_b_result = wf_b_evaluator.run()

        logger.info("Running Combined walk-forward (9 folds, fixed params)...")
        combined_evaluator = WalkForwardCombined(
            price_data=price_data,
            calendar=calendar,
            universe_members=members_fn,
            initial_capital=cfg.risk.initial_capital,
            base_engine_a_cfg=cfg.engine_a,
            base_engine_b_cfg=cfg.engine_b,
            risk_cfg=cfg.risk,
            cost_cfg=cfg.costs,
            wf_a_result=wf_a_result,
            wf_b_result=wf_b_result,
            sector_map=sector_map,
        )
        wf_result = combined_evaluator.run()
        n_trials = N_TRIALS_COMBINED
        engine_label = "Combined Engine"
        output_path = Path("reports") / "walk_forward_combined.md"

    else:
        logger.error("Unknown engine: %s", engine_name)
        return 1

    # Registry
    registry = ExperimentRegistry(Path("data/experiments.db"))
    exp_id = registry.log_run(
        engine=f"{engine_name.upper()}_walk_forward",
        start_date=start_date,
        end_date=end_date,
        params={"n_folds": 9, "n_trials": n_trials},
        metrics={
            "sharpe": wf_result.avg_test_sharpe,
            "cagr": wf_result.avg_test_cagr,
        },
    )
    for fold in wf_result.folds:
        registry.log_fold(
            experiment_id=exp_id,
            fold_index=fold.fold_index,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            train_sharpe=fold.train_sharpe,
            test_sharpe=fold.test_sharpe,
            test_cagr=fold.test_cagr,
            test_max_dd=fold.test_max_dd,
            best_params=fold.best_params,
        )
    registry.close()

    _print_and_write_wf_report(wf_result, engine_label, n_trials, output_path)
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
# Scanner
# ──────────────────────────────────────────────────────────────────────────────

def cmd_scanner_run(args) -> int:
    """Run the daily scanner, send Telegram alert, update status page."""
    from .scanner.run import DailyScanner, DEFAULT_STATE_PATH
    from .scanner.status_page import generate_status_page
    from .alerts.telegram import TelegramAlert, format_report

    config_dir = Path(args.config_dir)
    state_path = Path(args.state_path) if args.state_path else DEFAULT_STATE_PATH
    engine = args.engine
    dry_run: bool = args.dry_run

    scanner = DailyScanner(
        config_dir=config_dir,
        engine=engine,
        state_path=state_path,
        dry_run=dry_run,
    )

    report = scanner.scan()

    # Status page (always, even in dry-run)
    from .scanner.state import ScannerState
    state = ScannerState.load(state_path, initial_capital=report.initial_capital)
    generate_status_page(report, state)

    # Telegram alert
    if not dry_run:
        alert = TelegramAlert.from_env()
        if alert:
            ok = alert.send(format_report(report))
            if not ok:
                logger.warning("Telegram send failed.")
        else:
            logger.info("No Telegram credentials — skipping alert.")
    else:
        logger.info("[dry-run] Telegram alert skipped.")
        logger.info(format_report(report))

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

    # ── backtest ───────────────────────────────────────────────────────────────
    bt_p = sub.add_parser("backtest", help="Backtest commands")
    bt_sub = bt_p.add_subparsers(dest="subcommand", required=True)

    bt_run = bt_sub.add_parser("run", help="Run a backtest")
    bt_run.add_argument("--engine", default="a", choices=["a", "b", "combined"], metavar="ENGINE")
    bt_run.add_argument("--start", default="2005-01-01", metavar="YYYY-MM-DD")
    bt_run.add_argument("--end", default="2018-12-31", metavar="YYYY-MM-DD")
    bt_run.add_argument("--stress", action="store_true", help="Apply 2× cost multiplier")
    bt_run.set_defaults(func=cmd_backtest_run)

    bt_wf = bt_sub.add_parser("walk-forward", help="Run walk-forward evaluation")
    bt_wf.add_argument("--engine", default="a", choices=["a", "b", "combined"], metavar="ENGINE")
    bt_wf.set_defaults(func=cmd_backtest_walk_forward)

    # ── scanner ────────────────────────────────────────────────────────────────
    sc_p = sub.add_parser("scanner", help="Daily scanner commands")
    sc_sub = sc_p.add_subparsers(dest="subcommand", required=True)

    sc_run = sc_sub.add_parser("run", help="Run the daily scanner")
    sc_run.add_argument("--engine", default="a", choices=["a", "b", "combined"], metavar="ENGINE")
    sc_run.add_argument("--dry-run", action="store_true", help="Skip Telegram and state save")
    sc_run.add_argument("--state-path", default=None, metavar="PATH",
                        help="Override default state file path")
    sc_run.set_defaults(func=cmd_scanner_run)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
