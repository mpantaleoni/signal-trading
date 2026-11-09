"""Shared result types for backtest engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class RunResult:
    equity_curve: pd.DataFrame       # index=date, columns=["equity"]
    trades: list[dict]
    cash: float
    total_commission_usd: float
    total_tax_usd: float
    total_slippage_usd: float
    n_trades: int
    start_date: date
    end_date: date
    engine: str = "momentum"
