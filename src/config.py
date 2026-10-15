"""YAML configuration loader → typed dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    start_date: date
    end_date: date
    cache_dir: Path
    provider: str
    overlap_sessions: int = 10
    no_data_retry_days: int = 30
    adj_tolerance: float = 0.02
    large_return_threshold: float = 0.40
    stale_price_run: int = 5
    with_data_threshold: float = 0.90
    throttle_sleep: float = 0.5
    max_retries: int = 3
    retry_backoff: float = 2.0


@dataclass
class UniverseConfig:
    constituent_source: str
    constituent_sha256: str
    constituent_commit: str
    aliases: dict[str, str] = field(default_factory=dict)
    etfs: list[str] = field(default_factory=list)
    liquidity_min_adv_usd: float = 5_000_000
    earnings_blackout_enabled: bool = False
    earnings_blackout_days: int = 3


@dataclass
class CostConfig:
    commission_fixed: float = 12.95   # account currency per trade, one-way
    commission_pct: float = 0.0       # fraction of trade value
    slippage_bps: float = 5.0         # basis points per side
    stress_cost_multiplier: float = 2.0
    tax_rate: float = 0.26375
    tax_annual_allowance: float = 1000.0


@dataclass
class EngineAConfig:
    rsi_entry: int = 10
    max_hold_days: int = 10
    spy_symbol: str = "SPY"


@dataclass
class EngineBConfig:
    top_n: int = 10
    activity_lookback_fast: int = 21
    activity_lookback_slow: int = 252
    momentum_lookback: int = 21
    momentum_skip: int = 2
    spy_symbol: str = "SPY"


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.0075
    max_position_pct: float = 0.10
    max_positions_engine_a: int = 8
    max_positions_engine_b: int = 10
    max_positions_combined: int = 15
    sector_cap: int = 3
    drawdown_warning_pct: float = 0.10
    drawdown_stop_pct: float = 0.20
    atr_floor: float = 0.001
    min_history_days: int = 210
    initial_capital: float = 100_000.0


@dataclass
class Config:
    data: DataConfig
    universe: UniverseConfig
    costs: CostConfig = field(default_factory=CostConfig)
    engine_a: EngineAConfig = field(default_factory=EngineAConfig)
    engine_b: EngineBConfig = field(default_factory=EngineBConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


def load_config(config_dir: Path = Path("config")) -> Config:
    """Load config/*.yaml into a Config object."""
    data_path = config_dir / "data.yaml"
    universe_path = config_dir / "universe.yaml"
    costs_path = config_dir / "costs.yaml"
    engines_path = config_dir / "engines.yaml"
    risk_path = config_dir / "risk.yaml"

    with open(data_path) as f:
        d = yaml.safe_load(f)
    with open(universe_path) as f:
        u = yaml.safe_load(f)
    c: dict = {}
    if costs_path.exists():
        with open(costs_path) as f:
            c = yaml.safe_load(f) or {}
    e: dict = {}
    if engines_path.exists():
        with open(engines_path) as f:
            e = yaml.safe_load(f) or {}
    r: dict = {}
    if risk_path.exists():
        with open(risk_path) as f:
            r = yaml.safe_load(f) or {}

    end_raw = d.get("end_date", "today")
    end_date = date.today() if end_raw == "today" else date.fromisoformat(str(end_raw))

    data_cfg = DataConfig(
        start_date=date.fromisoformat(str(d["start_date"])),
        end_date=end_date,
        cache_dir=Path(d["cache_dir"]),
        provider=d["provider"],
        overlap_sessions=int(d.get("overlap_sessions", 10)),
        no_data_retry_days=int(d.get("no_data_retry_days", 30)),
        adj_tolerance=float(d.get("adj_tolerance", 0.02)),
        large_return_threshold=float(d.get("large_return_threshold", 0.40)),
        stale_price_run=int(d.get("stale_price_run", 5)),
        with_data_threshold=float(d.get("with_data_threshold", 0.90)),
        throttle_sleep=float(d.get("throttle_sleep", 0.5)),
        max_retries=int(d.get("max_retries", 3)),
        retry_backoff=float(d.get("retry_backoff", 2.0)),
    )

    universe_cfg = UniverseConfig(
        constituent_source=u.get("constituent_source", "data/raw/sp500_history.csv"),
        constituent_sha256=u.get("constituent_sha256", "") or "",
        constituent_commit=u.get("constituent_commit", "") or "",
        aliases={str(k): str(v) for k, v in (u.get("aliases") or {}).items()},
        etfs=list(u.get("etfs") or []),
        liquidity_min_adv_usd=float(u.get("liquidity_min_adv_usd", 5_000_000)),
        earnings_blackout_enabled=bool(u.get("earnings_blackout_enabled", False)),
        earnings_blackout_days=int(u.get("earnings_blackout_days", 3)),
    )

    costs_cfg = CostConfig(
        commission_fixed=float(c.get("commission_fixed", 12.95)),
        commission_pct=float(c.get("commission_pct", 0.0)),
        slippage_bps=float(c.get("slippage_bps", 5.0)),
        stress_cost_multiplier=float(c.get("stress_cost_multiplier", 2.0)),
        tax_rate=float(c.get("tax_rate", 0.26375)),
        tax_annual_allowance=float(c.get("tax_annual_allowance", 1000.0)),
    )

    ea_raw = e.get("engine_a", {}) or {}
    engine_a_cfg = EngineAConfig(
        rsi_entry=int(ea_raw.get("rsi_entry", 10)),
        max_hold_days=int(ea_raw.get("max_hold_days", 10)),
        spy_symbol=str(ea_raw.get("spy_symbol", "SPY")),
    )

    eb_raw = e.get("engine_b", {}) or {}
    engine_b_cfg = EngineBConfig(
        top_n=int(eb_raw.get("top_n", 10)),
        activity_lookback_fast=int(eb_raw.get("activity_lookback_fast", 21)),
        activity_lookback_slow=int(eb_raw.get("activity_lookback_slow", 252)),
        momentum_lookback=int(eb_raw.get("momentum_lookback", 21)),
        momentum_skip=int(eb_raw.get("momentum_skip", 2)),
        spy_symbol=str(eb_raw.get("spy_symbol", "SPY")),
    )

    risk_cfg = RiskConfig(
        risk_per_trade=float(r.get("risk_per_trade", 0.0075)),
        max_position_pct=float(r.get("max_position_pct", 0.10)),
        max_positions_engine_a=int(r.get("max_positions_engine_a", 8)),
        max_positions_engine_b=int(r.get("max_positions_engine_b", 10)),
        max_positions_combined=int(r.get("max_positions_combined", 15)),
        sector_cap=int(r.get("sector_cap", 3)),
        drawdown_warning_pct=float(r.get("drawdown_warning_pct", 0.10)),
        drawdown_stop_pct=float(r.get("drawdown_stop_pct", 0.20)),
        atr_floor=float(r.get("atr_floor", 0.001)),
        min_history_days=int(r.get("min_history_days", 210)),
        initial_capital=float(r.get("initial_capital", 100_000.0)),
    )

    return Config(data=data_cfg, universe=universe_cfg, costs=costs_cfg,
                  engine_a=engine_a_cfg, engine_b=engine_b_cfg, risk=risk_cfg)
