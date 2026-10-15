"""Position sizing — volatility-based (ATR) with equity fraction cap."""
from __future__ import annotations

import math


def compute_shares(
    equity: float,
    price: float,
    atr_value: float,
    risk_per_trade: float,
    max_position_pct: float,
    atr_floor: float = 0.001,
) -> int:
    """Compute number of shares to buy.

    Sizing rule: size so that a 2×ATR move = risk_per_trade × equity.
        raw_shares = floor(equity * risk_per_trade / (2 * atr_effective))

    Cap: no single position may exceed max_position_pct of equity.
        max_shares = floor(equity * max_position_pct / price)

    Returns min(raw_shares, max_shares), or 0 if the result is < 1.

    Parameters
    ----------
    equity         : current portfolio equity in USD
    price          : current price per share in USD
    atr_value      : ATR in USD (may be 0 or near-0 — floor is applied)
    risk_per_trade : fraction of equity risked per trade (e.g. 0.0075)
    max_position_pct: maximum fraction of equity in a single position (e.g. 0.10)
    atr_floor      : minimum ATR expressed as a fraction of price (prevents ÷0)
    """
    if equity <= 0 or price <= 0:
        return 0

    atr_effective = max(atr_value, price * atr_floor)
    raw_shares = math.floor(equity * risk_per_trade / (2.0 * atr_effective))

    max_shares = math.floor(equity * max_position_pct / price)

    result = min(raw_shares, max_shares)
    return result if result >= 1 else 0
