"""Tests for the click-based brain, economy (real numbers) and click sim."""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from bot.economy import HARD_LIMIT, SOFT_LIMIT, TroopTracker
from bot.planner import ClickPlanner, ClickPlannerConfig
from bot.vision import find_attack_targets, find_expand_targets
from sim.arena import build_state


# ---------- economy (real numbers from the doc) ----------

def test_economy_starts_at_512_balance_12_land():
    t = TroopTracker()
    assert t.balance == 512.0
    assert t.land == 12


def test_economy_hard_cap_enforced():
    t = TroopTracker(balance=10_000, land=12)
    for _ in range(50):
        t.update(land=12)
    assert t.balance <= HARD_LIMIT * 12 + 1


def test_economy_interest_zero_above_hard_limit():
    t = TroopTracker(balance=HARD_LIMIT * 100 + 1, land=100)
    t.update(land=100)
    # rate should be 0 at/above the hard limit
    assert t.rate() == 0.0


def test_economy_rate_decays_above_soft_limit():
    t = TroopTracker(balance=SOFT_LIMIT * 100, land=100)  # density = 100 → soft limit
    rate_at_soft = t.rate()
    t2 = TroopTracker(balance=SOFT_LIMIT * 135, land=100)  # density = 135
    assert t2.rate() < rate_at_soft


def test_economy_income_adds_land_every_10_ticks():
    t = TroopTracker(balance=0, land=500)
    for _ in range(9):
        t.update(land=500)
    assert t.balance == 0  # only interest so far (0 balance → 0)
    t.update(land=500)  # 10th tick → +land
    assert t.balance >= 500


def test_attack_budget_is_percent_of_balance():
    t = TroopTracker(balance=1000, land=100)
    assert t.attack_budget(10) == 100.0
    assert t.attack_budget(0) == 0.0


# ---------- vision: expand/attack targets ----------

def _flat_state(h, w, self_rect, enemy_rect):
    from bot.state import Blob, FrameState
    labels = np.zeros((h, w), dtype=int)
    y0, y1, x0, x1 = self_rect
    labels[y0:y1, x0:x1] = 1
    ey0, ey1, ex0, ex1 = enemy_rect
    labels[ey0:ey1, ex0:ex1] = 2
    me_mask = labels == 1
    mc = np.argwhere(me_mask)
    me = Blob("me", me_mask, len(mc), (float(mc[:, 0].mean()), float(mc[:, 1].mean())))
    em = labels == 2
    ec = np.argwhere(em)
    enemy = Blob("e2", em, len(ec), (float(ec[:, 0].mean()), float(ec[:, 1].mean())))
    expand = find_expand_targets(me_mask, labels == 0, max_samples=200)
    attack = find_attack_targets(me_mask, em, max_samples=200)
    return FrameState(shape=(h, w), labels=labels, self_blob=me, enemies=[enemy],
                      frontiers=np.zeros((0, 2), dtype=int),
                      expand_targets=expand, attack_targets=attack)


def test_expand_targets_are_neutral_adjacent_to_me():
    state = _flat_state(100, 100, (40, 60, 40, 60), (80, 90, 80, 90))
    assert len(state.expand_targets) > 0
    for y, x in state.expand_targets[:10]:
        assert state.labels[y, x] == 0  # neutral
        # adjacent to my territory
        near = state.labels[max(0, y-1):y+2, max(0, x-1):x+2]
        assert (near == 1).any()


def test_attack_targets_are_enemy_adjacent_to_me():
    # enemy shares a horizontal edge with me (overlapping columns)
    state = _flat_state(100, 100, (40, 60, 40, 60), (60, 75, 50, 65))
    assert len(state.attack_targets) > 0
    for y, x in state.attack_targets[:10]:
        assert state.labels[y, x] == 2  # enemy
        near = state.labels[max(0, y-1):y+2, max(0, x-1):x+2]
        assert (near == 1).any()


# ---------- ClickPlanner ----------

def test_clickplanner_expands_when_healthy():
    state = build_state(h=240, w=320, self_center=(120, 160), self_radius=25)
    planner = ClickPlanner()
    act = planner.decide(state)
    assert act.kind == "expand"
    assert act.x > 0 and act.y > 0


def test_clickplanner_banks_when_no_targets():
    # full-screen self territory → no neutral → bank
    h, w = 120, 160
    from bot.state import Blob, FrameState
    mask = np.ones((h, w), dtype=bool)
    me = Blob("me", mask, h * w, (h / 2, w / 2))
    state = FrameState(shape=(h, w), labels=np.ones((h, w), dtype=int),
                       self_blob=me, enemies=[],
                       expand_targets=np.zeros((0, 2), dtype=int),
                       attack_targets=np.zeros((0, 2), dtype=int))
    planner = ClickPlanner()
    act = planner.decide(state)
    assert act.kind == "bank"


def test_clickplanner_attacks_weak_neighbor():
    # small enemy ADJACENT to my big territory → attack
    state = build_state(h=240, w=320, self_center=(120, 160), self_radius=40,
                        enemies=[((120, 208), 10)])
    planner = ClickPlanner()
    act = planner.decide(state)
    assert act.kind == "attack"
    assert 0 < act.pct <= 20  # low slider %, per the doc's exploit heuristic


def test_clickplanner_spends_when_near_cap():
    state = build_state(h=240, w=320, self_center=(120, 160), self_radius=25)
    tracker = TroopTracker(balance=HARD_LIMIT * 25 * 0.9, land=25 * 25)
    planner = ClickPlanner(ClickPlannerConfig(), tracker)
    act = planner.decide(state)
    # near the hard cap it should still act (expand) rather than idle
    assert act.kind in ("expand", "attack")
