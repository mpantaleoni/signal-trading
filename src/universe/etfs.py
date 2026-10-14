"""ETF cross-check universe."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ETFUniverse:
    """A fixed list of liquid ETFs used for strategy cross-checking."""

    _symbols: list[str]

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = list(symbols)

    @classmethod
    def from_config(cls, etf_list: list[str]) -> "ETFUniverse":
        return cls(etf_list)

    def symbols(self) -> list[str]:
        return list(self._symbols)

    def __len__(self) -> int:
        return len(self._symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._symbols
