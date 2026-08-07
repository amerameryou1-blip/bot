#!/usr/bin/env python3
"""Train the bot's brain — evolutionary search over ClickPlanner weights.

Runs thousands of simulated matches (headless click sim, real mechanics) and
evolves the decision weights to maximize win rate (finishing #1 in territory)
and growth vs bot opponents. Saves the best weights to weights/best_weights.json.

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

# ---- parameter ranges (evolved) -------------------------------------------
PARAMS = {
    "expand_pct": (8.0, 20.0),
    "attack_pct": (2.0, 15.0),
    "attack_pct_capped": (2.0, 8.0),
    "weak_balance_ratio": (0.05, 0.6),
    "weak_area_ratio": (0.15, 0.7),
    "spend_density": (60.0, 120.0),
    "capped_density": (115.0, 148.0),
    "expand_radius": (8.0, 26.0),
}


def random_weights(rng) -> dict:
    w = {}
    for k, (lo, hi) in PARAMS.items():
        w[k] = round(rng.uniform(lo, hi), 3)
    return w


def mutate(w: dict, rng, rate=0.35, sigma=0.25) -> dict:
    out = {}
    for k, (lo, hi) in PARAMS.items():
        v = w[k]
        if rng.random() < rate:
            v += rng.gauss(0, sigma) * (hi - lo)
        out[k] = round(min(hi, max(lo, v)), 3)
    return out


def make_planner(w: dict) -> ClickPlanner:
    cfg = ClickPlannerConfig()
    for k, v in w.items():
        setattr(cfg, k, float(v))
    return ClickPlanner(cfg, TroopTracker(balance=512.0, land=12))


def evaluate(w: dict, seeds=(1, 2, 3, 4), n_bots=2, h=150, ww=210, max_ticks=1000, cpt=6) -> float:
    """Fitness = rank-1 rate + growth bonus. Higher is better."""
    rank1 = 0
    growth = 0
    for seed in seeds:
        game = ClickSim(h=h, w=ww, n_bots=n_bots, seed=seed, max_ticks=max_ticks, clicks_per_tick=cpt)
        planner = make_planner(w)
        r = game.run_match(planner.decide)
        areas = {pid: int((game.world == pid).sum()) for pid in game._pids if game.players[pid].alive}
        my = areas.get(1, 0)
        best_other = max((a for pid, a in areas.items() if pid != 1), default=0)
        if my >= best_other and my > 0:
            rank1 += 1
        growth += max(my - 12, 0)
    return rank1 / len(seeds) + growth / (len(seeds) * 4000.0)


def train(generations: int, population: int) -> dict:
    rng = random.Random(2026)
    pop = [random_weights(rng) for _ in range(population)]
    best_w, best_f = None, -1.0
    t0 = time.time()
    for gen in range(generations):
        scored = []
        for i, w in enumerate(pop):
            f = evaluate(w)
            scored.append((f, w))
            if f > best_f:
                best_f, best_w = f, dict(w)
        scored.sort(key=lambda x: -x[0])
        print(f"gen {gen+1}/{generations}: best fit={scored[0][0]:.3f} avg={sum(s[0] for s in scored)/len(scored):.3f} "
              f"best_weights={scored[0][1]} ({time.time()-t0:.0f}s)", flush=True)
        # keep top 3 + mutations + new randoms
        top = [dict(s[1]) for s in scored[:3]]
        pop = list(top)
        while len(pop) < population:
            parent = rng.choice(top)
            pop.append(mutate(parent, rng))
        if best_w:
            # always keep the overall best in the pool
            pop[0] = dict(best_w)
    return best_w


def main():
    generations = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    population = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"training: {generations} gens x {population} pop")
    best = train(generations, population)
    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_OUT.write_text(json.dumps(best, indent=2))
    print(f"\nBEST WEIGHTS -> {WEIGHTS_OUT}")
    print(json.dumps(best, indent=2))
    # final validation
    f = evaluate(best, seeds=(5, 6, 7, 8))
    print(f"validation fitness on held-out seeds: {f:.3f}")


if __name__ == "__main__":
    main()
