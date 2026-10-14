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


@dataclass
class Config:
    data: DataConfig
    universe: UniverseConfig


def load_config(config_dir: Path = Path("config")) -> Config:
    """Load config/data.yaml and config/universe.yaml into a Config object."""
    data_path = config_dir / "data.yaml"
    universe_path = config_dir / "universe.yaml"

    with open(data_path) as f:
        d = yaml.safe_load(f)
    with open(universe_path) as f:
        u = yaml.safe_load(f)

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
    )

    return Config(data=data_cfg, universe=universe_cfg)
