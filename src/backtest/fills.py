"""Fill price and commission calculations.

Currency note: Fineco charges €12.95 per trade (one-way). We convert to USD
using a fixed EUR/USD rate. This is a documented simplification — in practice
the rate fluctuates. See CLAUDE.md §5.
"""
from __future__ import annotations

EUR_USD_RATE: float = 1.10  # fixed EUR/USD conversion rate


def compute_fill_price(
    raw_price: float,
    side: str,
    slippage_bps: float,
    stress_multiplier: float = 1.0,
) -> float:
    """Apply slippage to a raw open price.

    Buys are filled slightly higher, sells slightly lower.

    Parameters
    ----------
    raw_price        : raw open price in USD
    side             : "buy" or "sell"
    slippage_bps     : slippage per side in basis points (e.g. 5.0)
    stress_multiplier: for stress-test mode, multiply slippage by this factor
    """
    effective_bps = slippage_bps * stress_multiplier
    slip = raw_price * effective_bps / 10_000.0
    if side == "buy":
        return raw_price + slip
    elif side == "sell":
        return raw_price - slip
    else:
        raise ValueError(f"Unknown side: {side!r}")


def compute_commission(
    commission_fixed_eur: float,
    commission_pct: float,
    trade_value_usd: float,
    stress_multiplier: float = 1.0,
) -> float:
    """Compute total commission in USD for one trade leg.

    Parameters
    ----------
    commission_fixed_eur : fixed fee in EUR (e.g. 12.95)
    commission_pct       : variable fraction of trade value (e.g. 0.0)
    trade_value_usd      : abs(shares × fill_price) in USD
    stress_multiplier    : multiply the entire commission for stress testing
    """
    fixed_usd = commission_fixed_eur * EUR_USD_RATE
    variable_usd = commission_pct * trade_value_usd
    total = (fixed_usd + variable_usd) * stress_multiplier
    return total
