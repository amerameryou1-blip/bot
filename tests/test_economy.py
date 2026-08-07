"""Tests for the troop economy tracker (mirrors territorial.io's model)."""
from __future__ import annotations

from bot.economy import TroopTracker


def test_interest_grows_troops_over_time() -> None:
    t = TroopTracker(initial_troops=1000)
    for _ in range(20):  # two interest payments
        t.update(land=1000)
    assert t.troops > 1000


def test_land_income_adds_land_every_100_ticks() -> None:
    t = TroopTracker(initial_troops=0)
    for _ in range(99):
        t.update(land=500)
    # At tick 99 (100th update) land income lands.
    assert t.troops >= 500


def test_claim_consumes_two_troops_per_pixel() -> None:
    t = TroopTracker(initial_troops=100)
    assert t.can_claim(50)
    claimed = t.claim(50)
    assert claimed == 50
    assert t.troops == 0
    assert not t.can_claim(1)


def test_claim_limited_by_troops() -> None:
    t = TroopTracker(initial_troops=9)
    claimed = t.claim(100)
    assert claimed == 4  # floor(9/2)
    assert t.troops == 1


def test_troop_cap_is_150x_land() -> None:
    t = TroopTracker(initial_troops=10_000_000)
    for _ in range(10):
        t.update(land=100)
    assert t.troops <= 150 * 100 + 1
