"""Tests for compute_shares edge cases."""
from __future__ import annotations

import pytest

from src.portfolio.sizing import compute_shares


class TestComputeShares:
    def test_basic(self):
        # equity=100k, price=50, atr=1.0, risk=0.75%, max_pos=10%
        # raw = floor(100000 * 0.0075 / (2 * 1.0)) = floor(375) = 375
        # max = floor(100000 * 0.10 / 50) = floor(200) = 200
        # result = min(375, 200) = 200
        shares = compute_shares(100_000, 50.0, 1.0, 0.0075, 0.10)
        assert shares == 200

    def test_raw_shares_limits(self):
        # equity=100k, price=5, atr=0.5, risk=0.75%, max_pos=10%
        # raw = floor(100000 * 0.0075 / (2 * 0.5)) = floor(750) = 750
        # max = floor(100000 * 0.10 / 5) = floor(2000) = 2000
        # result = min(750, 2000) = 750
        shares = compute_shares(100_000, 5.0, 0.5, 0.0075, 0.10)
        assert shares == 750

    def test_zero_atr_uses_floor(self):
        # atr=0 → atr_effective = price * atr_floor = 100 * 0.001 = 0.1
        # raw = floor(100000 * 0.0075 / (2 * 0.1)) = floor(3750) = 3750
        # max = floor(100000 * 0.10 / 100) = 100
        # result = min(3750, 100) = 100
        shares = compute_shares(100_000, 100.0, 0.0, 0.0075, 0.10, atr_floor=0.001)
        assert shares == 100

    def test_near_zero_atr_uses_floor(self):
        shares_zero = compute_shares(100_000, 100.0, 0.0, 0.0075, 0.10)
        shares_tiny = compute_shares(100_000, 100.0, 0.0001, 0.0075, 0.10)
        # Both should hit the cap at max_position_pct
        assert shares_zero == shares_tiny

    def test_returns_zero_when_less_than_one_share(self):
        # Tiny equity: can't afford even 1 share
        shares = compute_shares(10.0, 100.0, 1.0, 0.0075, 0.10)
        assert shares == 0

    def test_zero_equity_returns_zero(self):
        assert compute_shares(0.0, 50.0, 1.0, 0.0075, 0.10) == 0

    def test_zero_price_returns_zero(self):
        assert compute_shares(100_000, 0.0, 1.0, 0.0075, 0.10) == 0

    def test_min_one_share(self):
        # equity=100k, very high price, but should still give ≥1 if possible
        shares = compute_shares(100_000, 9_000.0, 10.0, 0.0075, 0.10)
        # raw = floor(100000 * 0.0075 / 20) = floor(37.5) = 37
        # max = floor(100000 * 0.10 / 9000) = floor(1.11) = 1
        assert shares == 1

    def test_cap_limits_position(self):
        # Large ATR → raw_shares is small, cap should not be the binding constraint
        shares = compute_shares(100_000, 50.0, 50.0, 0.0075, 0.10)
        # raw = floor(100000 * 0.0075 / 100) = floor(7.5) = 7
        # max = 200
        assert shares == 7

    def test_floor_parameter(self):
        # Custom atr_floor=0.01 → atr_effective = max(0, 100*0.01) = 1.0
        shares = compute_shares(100_000, 100.0, 0.0, 0.0075, 0.10, atr_floor=0.01)
        # raw = floor(100000 * 0.0075 / 2.0) = 375
        # max = floor(100000 * 0.10 / 100) = 100
        assert shares == 100
