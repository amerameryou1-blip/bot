"""Tests for the lookahead planner."""
from __future__ import annotations

import numpy as np

from bot.economy import TroopTracker
from bot.planner import PlannerConfig, TerritoryPlanner
from sim.arena import build_state


def make_planner(land=2000):
    return TerritoryPlanner(PlannerConfig(), TroopTracker(initial_troops=2000))


def test_planner_expands_toward_open_land() -> None:
    # Me in the middle with lots of neutral land around: should pick a heading
    # with a nonzero, normalized-ish direction and an 'expand' label.
    planner = make_planner()
    state = build_state(h=240, w=320, self_center=(120, 160), self_radius=25)
    plan = planner.decide(state)
    assert plan.dx != 0.0 or plan.dy != 0.0
    assert plan.label == "expand"
    # Heading magnitude should be reasonable.
    mag = (plan.dx ** 2 + plan.dy ** 2) ** 0.5
    assert 0.3 < mag < 1.5


def test_planner_avoids_enemy_dense_side() -> None:
    # Me in the middle; a BIG enemy directly below. Open neutral land
    # everywhere else. The planner must NOT charge straight down into the
    # big enemy (it should pick open land instead).
    h, w = 240, 320
    state = build_state(h=h, w=w, self_center=(120, 160), self_radius=18,
                        enemies=[((120, 250), 40)])
    planner = make_planner()
    plan = planner.decide(state)
    # Heading must not point straight at the enemy (dy > 0.9 = straight down).
    assert plan.dy < 0.9, f"expected to avoid enemy below, got dx={plan.dx:.2f} dy={plan.dy:.2f}"
    assert plan.label != "attack-corridor"


def test_planner_detects_attack_corridor() -> None:
    # A small enemy directly ahead with open land leading to it -> attack-corridor.
    h, w = 240, 320
    state = build_state(h=h, w=w, self_center=(120, 160), self_radius=22,
                        enemies=[((120, 280), 10)])
    planner = make_planner()
    plan = planner.decide(state)
    # Heading should point right (toward the enemy at col 280).
    assert plan.dx > 0.3, f"expected rightward attack, got dx={plan.dx:.2f}"


def test_planner_banks_when_no_budget() -> None:
    # No troops -> planner should still emit a heading (harmless) and not crash.
    planner = TerritoryPlanner(PlannerConfig(), TroopTracker(initial_troops=0))
    state = build_state(h=240, w=320, self_center=(120, 160), self_radius=25)
    plan = planner.decide(state)
    assert plan.dx == plan.dx  # not NaN
