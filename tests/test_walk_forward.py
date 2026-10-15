"""Tests for WalkForwardEvaluator — fold generation (count, dates, non-overlap)."""
from __future__ import annotations

from datetime import date

import pytest

from src.backtest.walk_forward import N_TRIALS, _make_folds


class TestMakeFolds:
    def test_nine_folds(self):
        folds = _make_folds()
        assert len(folds) == 9

    def test_first_fold(self):
        folds = _make_folds()
        ts, te, vs, ve = folds[0]
        assert ts == date(2005, 1, 1)
        assert te == date(2009, 12, 31)
        assert vs == date(2010, 1, 1)
        assert ve == date(2010, 12, 31)

    def test_last_fold(self):
        folds = _make_folds()
        ts, te, vs, ve = folds[-1]
        assert ts == date(2013, 1, 1)
        assert te == date(2017, 12, 31)
        assert vs == date(2018, 1, 1)
        assert ve == date(2018, 12, 31)

    def test_no_train_test_overlap(self):
        folds = _make_folds()
        for ts, te, vs, ve in folds:
            assert te < vs, f"Train end {te} overlaps test start {vs}"

    def test_test_periods_cover_2010_to_2018(self):
        folds = _make_folds()
        test_years = [vs.year for _, _, vs, _ in folds]
        assert test_years == list(range(2010, 2019))

    def test_train_always_5_years(self):
        folds = _make_folds()
        for ts, te, vs, ve in folds:
            assert te.year - ts.year == 4  # 5 years span (2005–2009 = 5 years)

    def test_rolling_by_one_year(self):
        folds = _make_folds()
        for i in range(1, len(folds)):
            prev_ts = folds[i - 1][0]
            curr_ts = folds[i][0]
            assert curr_ts.year == prev_ts.year + 1

    def test_n_trials_is_36(self):
        # 4 param combos × 9 folds
        assert N_TRIALS == 36
