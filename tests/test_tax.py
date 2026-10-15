"""Tests for GermanTaxLedger — annual tax, carry-forward, allowance, multi-year."""
from __future__ import annotations

import pytest

from src.backtest.fills import EUR_USD_RATE
from src.backtest.tax import GermanTaxLedger


TAX_RATE = 0.26375
ALLOWANCE = 1000.0


class TestGermanTaxLedger:
    def test_no_gain_no_tax(self):
        ledger = GermanTaxLedger(TAX_RATE, ALLOWANCE)
        ledger.record_trade(0.0)
        tax = ledger.year_end()
        assert tax == pytest.approx(0.0)

    def test_gain_within_allowance_no_tax(self):
        ledger = GermanTaxLedger(TAX_RATE, ALLOWANCE)
        # Record €800 gain (< €1000 allowance)
        ledger.record_trade(800 * EUR_USD_RATE)
        tax = ledger.year_end()
        assert tax == pytest.approx(0.0)

    def test_gain_above_allowance_taxed(self):
        ledger = GermanTaxLedger(TAX_RATE, ALLOWANCE)
        # Record €2000 gain → taxable = 2000 - 1000 = 1000 EUR
        ledger.record_trade(2000 * EUR_USD_RATE)
        tax = ledger.year_end()
        expected_usd = 1000 * TAX_RATE * EUR_USD_RATE
        assert tax == pytest.approx(expected_usd)

    def test_loss_creates_carry_forward(self):
        ledger = GermanTaxLedger(TAX_RATE, ALLOWANCE)
        ledger.record_trade(-500 * EUR_USD_RATE)
        tax = ledger.year_end()
        assert tax == pytest.approx(0.0)
        assert ledger.carry_forward_eur == pytest.approx(500.0)

    def test_carry_forward_reduces_taxable_next_year(self):
        ledger = GermanTaxLedger(TAX_RATE, ALLOWANCE)
        # Year 1: loss of €800
        ledger.record_trade(-800 * EUR_USD_RATE)
        ledger.year_end()
        assert ledger.carry_forward_eur == pytest.approx(800.0)

        # Year 2: gain of €2000 → net = 2000 - 800 = 1200 → taxable = 1200 - 1000 = 200
        ledger.record_trade(2000 * EUR_USD_RATE)
        tax = ledger.year_end()
        expected_usd = 200 * TAX_RATE * EUR_USD_RATE
        assert tax == pytest.approx(expected_usd)
        assert ledger.carry_forward_eur == pytest.approx(0.0)

    def test_carry_forward_indefinite(self):
        ledger = GermanTaxLedger(TAX_RATE, ALLOWANCE)
        # Three loss years
        for _ in range(3):
            ledger.record_trade(-1000 * EUR_USD_RATE)
            ledger.year_end()
        assert ledger.carry_forward_eur == pytest.approx(3000.0)

    def test_multiple_trades_in_year(self):
        ledger = GermanTaxLedger(TAX_RATE, ALLOWANCE)
        ledger.record_trade(1000 * EUR_USD_RATE)
        ledger.record_trade(-200 * EUR_USD_RATE)
        ledger.record_trade(500 * EUR_USD_RATE)
        # Net = 1300 EUR → taxable = 1300 - 1000 = 300
        tax = ledger.year_end()
        expected_usd = 300 * TAX_RATE * EUR_USD_RATE
        assert tax == pytest.approx(expected_usd)

    def test_year_end_resets_current_year_pnl(self):
        ledger = GermanTaxLedger(TAX_RATE, ALLOWANCE)
        ledger.record_trade(5000 * EUR_USD_RATE)
        ledger.year_end()
        assert ledger.current_year_pnl_eur == pytest.approx(0.0)

    def test_carry_forward_cleared_after_net_gain(self):
        ledger = GermanTaxLedger(TAX_RATE, ALLOWANCE)
        ledger.record_trade(-500 * EUR_USD_RATE)
        ledger.year_end()  # carry_forward = 500
        ledger.record_trade(2000 * EUR_USD_RATE)
        ledger.year_end()
        assert ledger.carry_forward_eur == pytest.approx(0.0)

    def test_exact_allowance_no_tax(self):
        ledger = GermanTaxLedger(TAX_RATE, ALLOWANCE)
        ledger.record_trade(ALLOWANCE * EUR_USD_RATE)
        tax = ledger.year_end()
        assert tax == pytest.approx(0.0)
