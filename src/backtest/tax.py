"""German capital-gains tax ledger (Abgeltungsteuer).

Applied annually at year-end in the backtest. Losses carry forward
indefinitely. The allowance (Sparerpauschbetrag) is deducted before tax.

EUR/USD conversion uses the fixed rate from fills.py (documented limitation).
"""
from __future__ import annotations

from .fills import EUR_USD_RATE


class GermanTaxLedger:
    """Tracks realised P&L and computes annual tax liability.

    All monetary values are tracked in EUR. Tax is returned in USD
    (converted at the fixed EUR/USD rate).
    """

    def __init__(
        self,
        tax_rate: float = 0.26375,
        annual_allowance_eur: float = 1000.0,
    ) -> None:
        self._tax_rate = tax_rate
        self._allowance = annual_allowance_eur
        self._carry_forward_eur: float = 0.0
        self._current_year_pnl_eur: float = 0.0

    # ── recording ─────────────────────────────────────────────────────────────

    def record_trade(self, realized_pnl_usd: float) -> None:
        """Add a realised P&L from a closed trade (in USD)."""
        self._current_year_pnl_eur += realized_pnl_usd / EUR_USD_RATE

    # ── year-end settlement ───────────────────────────────────────────────────

    def year_end(self) -> float:
        """Compute and deduct annual tax. Returns tax paid in USD.

        Steps:
        1. net_eur = current_year_pnl - carry_forward
        2. If net_eur <= 0: add to carry_forward, return 0.
        3. taxable_eur = max(0, net_eur - annual_allowance)
        4. tax_eur = taxable_eur * tax_rate
        5. Reset current-year accumulator; carry_forward unchanged if net > 0.
        """
        net_eur = self._current_year_pnl_eur - self._carry_forward_eur

        if net_eur <= 0:
            self._carry_forward_eur = -net_eur  # losses grow carry-forward
            self._current_year_pnl_eur = 0.0
            return 0.0

        # Net positive — apply allowance
        taxable_eur = max(0.0, net_eur - self._allowance)
        tax_eur = taxable_eur * self._tax_rate

        # After netting out carry_forward, reset it to 0
        self._carry_forward_eur = 0.0
        self._current_year_pnl_eur = 0.0

        return tax_eur * EUR_USD_RATE

    # ── inspection ────────────────────────────────────────────────────────────

    @property
    def carry_forward_eur(self) -> float:
        return self._carry_forward_eur

    @property
    def current_year_pnl_eur(self) -> float:
        return self._current_year_pnl_eur
