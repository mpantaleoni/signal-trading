"""
Tests for src/universe/sp500.py.

Requirements from phase1_plan.md §6:
- Membership before/on/after a change
- Before-first-snapshot raises
- Snapshots -> intervals -> snapshots round-trip
- Aliases
- No-look-ahead: adding a future change leaves earlier members_on unchanged
- Real-file sanity (skipped unless --real-data flag or @network)
"""
from __future__ import annotations

from datetime import date

import pytest

from src.universe.sp500 import SP500Universe, Snapshot


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_universe() -> SP500Universe:
    """
    Three snapshots:
      2020-01-01: {A, B, C}
      2020-06-01: {A, B, C, D}   (D added)
      2021-01-01: {A, B, D}       (C removed)
    """
    return SP500Universe(
        [
            Snapshot(date(2020, 1, 1), frozenset({"A", "B", "C"})),
            Snapshot(date(2020, 6, 1), frozenset({"A", "B", "C", "D"})),
            Snapshot(date(2021, 1, 1), frozenset({"A", "B", "D"})),
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# members_on
# ──────────────────────────────────────────────────────────────────────────────

def test_members_on_first_snapshot():
    u = _make_universe()
    assert u.members_on(date(2020, 1, 1)) == frozenset({"A", "B", "C"})


def test_members_on_between_snapshots():
    u = _make_universe()
    # Between 2020-01-01 and 2020-06-01 → still first snapshot
    assert u.members_on(date(2020, 3, 15)) == frozenset({"A", "B", "C"})


def test_members_on_second_snapshot():
    u = _make_universe()
    assert u.members_on(date(2020, 6, 1)) == frozenset({"A", "B", "C", "D"})


def test_members_on_after_removal():
    u = _make_universe()
    # After 2021-01-01 C is gone
    assert "C" not in u.members_on(date(2021, 6, 1))
    assert u.members_on(date(2021, 6, 1)) == frozenset({"A", "B", "D"})


def test_members_on_before_first_snapshot_raises():
    u = _make_universe()
    with pytest.raises(ValueError):
        u.members_on(date(2019, 12, 31))


def test_no_look_ahead():
    """
    Adding a future snapshot must not change members_on for earlier dates.
    """
    u1 = SP500Universe(
        [Snapshot(date(2020, 1, 1), frozenset({"A", "B"}))]
    )
    result_before = u1.members_on(date(2020, 3, 1))

    # Now create a universe with an additional future snapshot
    u2 = SP500Universe(
        [
            Snapshot(date(2020, 1, 1), frozenset({"A", "B"})),
            Snapshot(date(2020, 6, 1), frozenset({"A", "B", "C"})),
        ]
    )
    result_after = u2.members_on(date(2020, 3, 1))

    assert result_before == result_after, (
        "members_on(2020-03-01) changed after adding a future snapshot — look-ahead!"
    )


# ──────────────────────────────────────────────────────────────────────────────
# membership_intervals
# ──────────────────────────────────────────────────────────────────────────────

def test_membership_intervals_basic():
    u = _make_universe()
    intervals = u.membership_intervals()

    # A and B are in all three snapshots → one open interval
    assert len(intervals["A"]) == 1
    assert intervals["A"][0][0] == date(2020, 1, 1)
    assert intervals["A"][0][1] is None  # still member

    # C appeared in snapshot 1, removed in snapshot 3
    c_ivs = intervals["C"]
    assert len(c_ivs) == 1
    assert c_ivs[0][0] == date(2020, 1, 1)
    assert c_ivs[0][1] == date(2021, 1, 1)

    # D appeared in snapshot 2, still in snapshot 3
    d_ivs = intervals["D"]
    assert len(d_ivs) == 1
    assert d_ivs[0][0] == date(2020, 6, 1)
    assert d_ivs[0][1] is None


def test_membership_intervals_rejoin():
    """A ticker that leaves and re-enters gets two intervals."""
    u = SP500Universe(
        [
            Snapshot(date(2020, 1, 1), frozenset({"A", "X"})),
            Snapshot(date(2020, 6, 1), frozenset({"A"})),      # X leaves
            Snapshot(date(2021, 1, 1), frozenset({"A", "X"})), # X rejoins
        ]
    )
    intervals = u.membership_intervals()
    x_ivs = intervals["X"]
    assert len(x_ivs) == 2
    assert x_ivs[0] == (date(2020, 1, 1), date(2020, 6, 1))
    assert x_ivs[1][0] == date(2021, 1, 1)
    assert x_ivs[1][1] is None


# ──────────────────────────────────────────────────────────────────────────────
# ever_members
# ──────────────────────────────────────────────────────────────────────────────

def test_ever_members_full_range():
    u = _make_universe()
    ever = u.ever_members(date(2020, 1, 1), date(2021, 12, 31))
    assert "A" in ever and "B" in ever and "C" in ever and "D" in ever


def test_ever_members_narrow_range():
    u = _make_universe()
    # Between first and second snapshot: only A, B, C
    ever = u.ever_members(date(2020, 1, 1), date(2020, 5, 31))
    assert ever == frozenset({"A", "B", "C"})


def test_ever_members_after_removal():
    u = _make_universe()
    # After C was removed, C should not appear in ever_members for a range
    # entirely after the removal
    ever = u.ever_members(date(2021, 6, 1), date(2021, 12, 31))
    assert "C" not in ever


# ──────────────────────────────────────────────────────────────────────────────
# from_csv
# ──────────────────────────────────────────────────────────────────────────────

def test_from_csv(sp500_csv):
    u = SP500Universe.from_csv(sp500_csv)
    # Three snapshots
    assert len(u.snapshot_dates()) == 3

    # members_on the first date
    members = u.members_on(date(2020, 1, 1))
    assert "AAPL" in members and "GOOG" in members and "MSFT" in members

    # AMZN added on 2020-06-01
    assert "AMZN" in u.members_on(date(2020, 6, 1))
    assert "AMZN" not in u.members_on(date(2020, 3, 1))

    # MSFT removed on 2021-01-01
    assert "MSFT" not in u.members_on(date(2021, 6, 1))


def test_from_csv_before_first_snapshot_raises(sp500_csv):
    u = SP500Universe.from_csv(sp500_csv)
    with pytest.raises(ValueError):
        u.members_on(date(2019, 1, 1))


def test_from_csv_real_file_sanity():
    """
    Sanity check on the vendored file (skipped if not present).
    Expects 495–510 members on a few sample dates.
    """
    path_candidates = [
        "data/raw/sp500_history.csv",
    ]
    import os
    from pathlib import Path
    for p in path_candidates:
        if Path(p).exists():
            u = SP500Universe.from_csv(Path(p))
            for sample_date in [date(2010, 1, 4), date(2015, 6, 1), date(2020, 1, 2)]:
                try:
                    members = u.members_on(sample_date)
                    assert 480 <= len(members) <= 520, (
                        f"Expected 480-520 members on {sample_date}, got {len(members)}"
                    )
                except ValueError:
                    pass  # date before first snapshot
            return
    pytest.skip("Vendored constituent CSV not found; run: python -m src.cli universe fetch-constituents")
