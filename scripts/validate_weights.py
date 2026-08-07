#!/usr/bin/env python3
"""Validate the trained weights: tournament vs bots, report rank-1 rate."""
import json, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sim.game4 import ClickSim
from bot.planner import ClickPlanner, ClickPlannerConfig
from bot.economy import TroopTracker

REPO = Path(__file__).resolve().parents[1]
weights = json.loads((REPO / "weights" / "best_weights.json").read_text())

def make_planner():
    cfg = ClickPlannerConfig()
    for k, v in weights.items():
        setattr(cfg, k, float(v))
    return ClickPlanner(cfg, TroopTracker(balance=512.0, land=12))

def run(seeds=8, n_bots=3):
    wins = 0
    total_growth = 0
    for seed in range(1, seeds + 1):
        game = ClickSim(h=180, w=250, n_bots=n_bots, seed=seed, max_ticks=2000, clicks_per_tick=10)
        r = game.run_match(make_planner().decide)
        areas = {pid: int((game.world == pid).sum()) for pid in game._pids if game.players[pid].alive}
        my = areas.get(1, 0)
        best_other = max((a for pid, a in areas.items() if pid != 1), default=0)
        rank1 = my >= best_other and my > 0
        if rank1:
            wins += 1
        total_growth += max(my - 12, 0)
        print(f"  seed {seed}: my={my} best_other={best_other} rank1={rank1}")
    print(f"\nTRAINED-WEIGHTS TOURNAMENT: #1 in {wins}/{seeds} matches, avg_growth={total_growth//seeds}px")

if __name__ == "__main__":
    run()
