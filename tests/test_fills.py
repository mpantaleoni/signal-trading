"""Tests for compute_fill_price and compute_commission."""
from __future__ import annotations

import pytest

from src.backtest.fills import EUR_USD_RATE, compute_commission, compute_fill_price


class TestComputeFillPrice:
    def test_buy_is_higher_than_raw(self):
        raw = 100.0
        fill = compute_fill_price(raw, "buy", slippage_bps=5.0)
        assert fill > raw

    def test_sell_is_lower_than_raw(self):
        raw = 100.0
        fill = compute_fill_price(raw, "sell", slippage_bps=5.0)
        assert fill < raw

    def test_buy_slippage_magnitude(self):
        raw = 100.0
        fill = compute_fill_price(raw, "buy", slippage_bps=5.0)
        assert fill == pytest.approx(100.0 * (1 + 5 / 10_000))

    def test_sell_slippage_magnitude(self):
        raw = 100.0
        fill = compute_fill_price(raw, "sell", slippage_bps=5.0)
        assert fill == pytest.approx(100.0 * (1 - 5 / 10_000))

    def test_stress_multiplier_doubles_slippage(self):
        raw = 100.0
        normal = compute_fill_price(raw, "buy", slippage_bps=5.0, stress_multiplier=1.0)
        stress = compute_fill_price(raw, "buy", slippage_bps=5.0, stress_multiplier=2.0)
        # Additional slippage should be double
        normal_slip = normal - raw
        stress_slip = stress - raw
        assert stress_slip == pytest.approx(2 * normal_slip)

    def test_zero_slippage(self):
        raw = 50.0
        assert compute_fill_price(raw, "buy", 0.0) == pytest.approx(raw)
        assert compute_fill_price(raw, "sell", 0.0) == pytest.approx(raw)

    def test_unknown_side_raises(self):
        with pytest.raises(ValueError):
            compute_fill_price(100.0, "hold", 5.0)


class TestComputeCommission:
    def test_fixed_only(self):
        comm = compute_commission(12.95, 0.0, 5000.0)
        assert comm == pytest.approx(12.95 * EUR_USD_RATE)

    def test_fixed_plus_variable(self):
        comm = compute_commission(12.95, 0.001, 10_000.0)
        expected = (12.95 * EUR_USD_RATE) + (0.001 * 10_000.0)
        assert comm == pytest.approx(expected)

    def test_stress_multiplier(self):
        base = compute_commission(12.95, 0.0, 5000.0, stress_multiplier=1.0)
        stress = compute_commission(12.95, 0.0, 5000.0, stress_multiplier=2.0)
        assert stress == pytest.approx(2 * base)

    def test_zero_commission(self):
        assert compute_commission(0.0, 0.0, 10_000.0) == pytest.approx(0.0)

    def test_eur_usd_conversion(self):
        # 1 EUR should become 1 * EUR_USD_RATE USD
        comm = compute_commission(1.0, 0.0, 0.0)
        assert comm == pytest.approx(EUR_USD_RATE)
