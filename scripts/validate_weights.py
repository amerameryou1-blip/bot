#!/usr/bin/env python3
"""Validate the trained combat brain: tournament vs bots, report LAST-SURVIVOR wins."""
import json, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sim.game4 import ClickSim
from bot.planner import ClickPlanner, ClickPlannerConfig
from bot.economy import TroopTracker

REPO = Path(__file__).resolve().parents[1]
weights = json.loads((REPO / "weights" / "best_weights.json").read_text())


def make_brain():
    cfg = ClickPlannerConfig()
    for k, v in weights.items():
        setattr(cfg, k, float(v))
    planner = ClickPlanner(cfg, TroopTracker(balance=512.0, land=12))

    def decide(state):
        return planner.decide(state)

    decide.planner = planner
    return decide


def run(seeds=8, n_bots=3):
    wins = 0
    total_ticks = 0
    for seed in range(1, seeds + 1):
        game = ClickSim(h=100, w=140, n_bots=n_bots, seed=seed, max_ticks=2000, clicks_per_tick=12)
        r = game.run_match(make_brain())
        survivor = r["winner"] == 1
        if survivor:
            wins += 1
        total_ticks += r["ticks"]
        print(f"  seed {seed}: winner={r['winner']} (we {'WIN' if survivor else 'lost'}) "
              f"our_max={r['our_max_area']} final={r['our_final_area']} ticks={r['ticks']} alive={r['alive']}")
    print(f"\nCOMBAT TOURNAMENT: LAST-SURVIVOR wins {wins}/{seeds}, avg match {total_ticks//seeds} ticks")


if __name__ == "__main__":
    run()
