#!/usr/bin/env python3
"""Validate the attack-meta planner in sim v5: last-survivor win + rank."""
import json, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sim.game5 import ClickSim5
from bot.planner import ClickPlanner, ClickPlannerConfig
from bot.economy import TroopTracker

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "best_weights.json"


def make_planner():
    cfg = ClickPlannerConfig()
    # keep the evolved expand_radius; meta attack params come from the class
    try:
        w = json.loads(WEIGHTS.read_text())
        if "expand_radius" in w:
            cfg.expand_radius = int(w["expand_radius"])
    except Exception:
        pass
    return ClickPlanner(cfg, TroopTracker(balance=512.0, land=12))


def run(seeds=8, n_bots=3):
    wins = 0
    rank_sum = 0
    for seed in range(1, seeds + 1):
        game = ClickSim5(h=180, w=250, n_bots=n_bots, seed=seed, max_ticks=2200, clicks_per_tick=10)
        r = game.run_match(make_planner().decide)
        wins += 1 if r["winner"] == 1 else 0
        rank_sum += r["our_rank"]
        print(f"  seed {seed}: winner={r['winner']} our_rank={r['our_rank']} our_max={r['our_max_area']} "
              f"ticks={r['ticks']} alive={r['alive']}", flush=True)
    print(f"\nMETA TEACHER: {wins}/{seeds} last-survivor wins, avg_rank={rank_sum/seeds:.2f}")


if __name__ == "__main__":
    run()
