"""Sanity tests for the headless simulator."""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from bot.planner import PlannerConfig, TerritoryPlanner
from bot.economy import TroopTracker
from sim.game3 import SimGame3


def _brain():
    tracker = TroopTracker(initial_troops=500)
    planner = TerritoryPlanner(PlannerConfig(risk_weight=1.6, attack_ratio_max=0.8,
                                             gain_weight=1.2, corner_penalty=0.2))
    cache = {"n": 0, "res": None}

    def decide(state):
        cache["n"] += 1
        if cache["res"] is None or cache["n"] % 5 == 0:
            cache["res"] = planner.decide(state)
        return cache["res"]

    return decide


def test_match_runs_and_resolves() -> None:
    game = SimGame3(h=160, w=220, n_bots=2, seed=1, max_ticks=800)
    result = game.run_match(_brain())
    assert result["ticks"] > 0
    assert result["winner"] in (0, 1, 2, 3)
    # somebody won or the cap was hit with someone alive
    assert len(result["alive"]) >= 1


def test_our_area_can_grow() -> None:
    game = SimGame3(h=160, w=220, n_bots=2, seed=2, max_ticks=600)
    result = game.run_match(_brain())
    assert result["our_max_area"] > 0


def test_sim_runs_deterministically_per_seed() -> None:
    r1 = SimGame3(h=160, w=220, n_bots=2, seed=5, max_ticks=400).run_match(_brain())
    r2 = SimGame3(h=160, w=220, n_bots=2, seed=5, max_ticks=400).run_match(_brain())
    assert r1["ticks"] == r2["ticks"]
    assert r1["our_max_area"] == r2["our_max_area"]
