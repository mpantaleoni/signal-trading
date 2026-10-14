"""Ticker normalisation and alias resolution."""
from __future__ import annotations


def normalize_yfinance(symbol: str) -> str:
    """
    Convert a constituent-list ticker to a yfinance-compatible symbol.

    Examples: BRK.B → BRK-B,  BF.B → BF-B
    yfinance uses hyphens where the exchange uses dots for class shares.
    """
    return symbol.replace(".", "-")


def normalize_stooq(symbol: str) -> str:
    """
    Convert a symbol to Stooq's format: lower-case + '.us' suffix.

    Examples: AAPL → aapl.us,  BRK-B → brk-b.us
    """
    return symbol.lower().replace(".", "-") + ".us"


def apply_alias(symbol: str, aliases: dict[str, str]) -> str:
    """
    Look up a historical ticker in the alias map.

    If the symbol is known (e.g. FB→META) return the current symbol;
    otherwise return the symbol unchanged.
    """
    return aliases.get(symbol, symbol)


def resolve_symbol(symbol: str, aliases: dict[str, str]) -> str:
    """
    Full resolution pipeline: apply alias, then normalize for yfinance.

    This is what the data layer uses before every fetch call.
    """
    aliased = apply_alias(symbol, aliases)
    return normalize_yfinance(aliased)
