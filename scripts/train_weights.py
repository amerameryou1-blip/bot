#!/usr/bin/env python3
"""Train the bot's combat brain — evolutionary search for LAST-SURVIVOR wins.

Simulates full matches (expand -> attack -> eliminate) and evolves the
ClickPlanner weights so the bot finishes as the LAST SURVIVOR vs bots.

Run:  PYTHONPATH=src python3 scripts/train_weights.py [generations] [population]
"""
import json, os, sys, time, random, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sim.game4 import ClickSim
from bot.planner import ClickPlanner, ClickPlannerConfig
from bot.economy import TroopTracker

REPO = Path(__file__).resolve().parents[1]
WEIGHTS_OUT = REPO / "weights" / "best_weights.json"

# ---- parameters to evolve (ranges) ----------------------------------------
PARAMS = {
    "expand_pct": (8.0, 20.0),
    "attack_pct": (5.0, 25.0),
    "weak_balance_ratio": (0.05, 0.5),
    "attack_balance_ratio": (1.2, 3.0),
    "attack_density": (40.0, 110.0),
    "spend_density": (70.0, 120.0),
    "capped_density": (115.0, 148.0),
    "expand_radius": (8.0, 26.0),
}


def random_weights(rng) -> dict:
    return {k: round(rng.uniform(lo, hi), 3) for k, (lo, hi) in PARAMS.items()}


def mutate(w: dict, rng, rate=0.35, sigma=0.25) -> dict:
    out = {}
    for k, (lo, hi) in PARAMS.items():
        v = w[k]
        if rng.random() < rate:
            v += rng.gauss(0, sigma) * (hi - lo)
        out[k] = round(min(hi, max(lo, v)), 3)
    return out


def make_brain(w: dict):
    cfg = ClickPlannerConfig()
    for k, v in w.items():
        setattr(cfg, k, float(v))
    planner = ClickPlanner(cfg, TroopTracker(balance=512.0, land=12))

    def decide(state):
        return planner.decide(state)

    decide.planner = planner  # sim feeds enemy balances via this
    return decide


def evaluate(w: dict, seeds=(1, 2, 3), n_bots=2, h=90, ww=125, max_ticks=2000) -> float:
    """Fitness = LAST-SURVIVOR win rate + growth bonus."""
    wins = 0
    growth = 0
    for seed in seeds:
        game = ClickSim(h=h, w=ww, n_bots=n_bots, seed=seed, max_ticks=max_ticks, clicks_per_tick=6)
        r = game.run_match(make_brain(w))
        if r["winner"] == 1:
            wins += 1
        growth += max(r["our_max_area"] - 12, 0)
    return wins / len(seeds) + growth / (len(seeds) * 5000.0)


def train(generations: int, population: int) -> dict:
    rng = random.Random(2026)
    pop = [random_weights(rng) for _ in range(population)]
    best_w, best_f = None, -1.0
    t0 = time.time()
    for gen in range(generations):
        scored = []
        for w in pop:
            f = evaluate(w)
            scored.append((f, w))
            if f > best_f:
                best_f, best_w = f, dict(w)
        scored.sort(key=lambda x: -x[0])
        print(f"gen {gen+1}/{generations}: best={scored[0][0]:.3f} avg={sum(s[0] for s in scored)/len(scored):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        top = [dict(s[1]) for s in scored[:3]]
        pop = list(top)
        while len(pop) < population:
            pop.append(mutate(rng.choice(top), rng))
        pop[0] = dict(best_w)
    return best_w


def main():
    generations = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    population = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(f"training combat brain: {generations} gens x {population} pop (fitness = last-survivor win rate)")
    best = train(generations, population)
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_OUT.write_text(json.dumps(best, indent=2))
    print(f"\nBEST WEIGHTS -> {WEIGHTS_OUT}")
    print(json.dumps(best, indent=2))
    f = evaluate(best, seeds=(5, 6, 7, 8))
    print(f"held-out validation fitness: {f:.3f}")


if __name__ == "__main__":
    main()
