#!/usr/bin/env python3
"""evolve_weights.py — DIY AlphaEvolve-style loop over the Stage-A weighting
formula (the one pure function where code-search pays). PLANNED for AFTER
distillation; do not divert GPU/CPU before then.

Loop (mirrors AlphaEvolve: propose -> evaluate -> keep fittest):
  1. candidates.json holds the population: list of
     {"survivor_mult":s, "kill_mult":k, "tau":t, "win_bonus":w, "note":...}
  2. Each candidate is scored by a CHEAP proxy: 2M farmer, 12 PPO steps on
     the same shard bundle + short capped-lobby eval (alive-rate + rank).
  3. Top-2 survive; the agents (B + A) act as mutation engine and write the
     next generation's candidates (4-6 per gen), informed by the scores.
  4. Champion confirmed ONLY by the honest 16-seed last-survivor eval.

Usage (Kaggle CPU kernel or beefy box):
  python3 scripts/evolve_weights.py --gen 0 --cands candidates.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_POP = [
    {"survivor_mult": 4.0, "kill_mult": 2.0, "tau": 1.0, "win_bonus": 5.0,
     "note": "current standard"},
    {"survivor_mult": 2.0, "kill_mult": 4.0, "tau": 1.0, "win_bonus": 5.0,
     "note": "aggressive arm"},
    {"survivor_mult": 3.0, "kill_mult": 3.0, "tau": 0.7, "win_bonus": 5.0,
     "note": "sharper AWR"},
    {"survivor_mult": 4.0, "kill_mult": 2.0, "tau": 1.0, "win_bonus": 8.0,
     "note": "win-hungry"},
]


# EVOLVE-BLOCK-START
def apply_weights(cand):
    """Inject a candidate into sovereign_data's AWR constants."""
    import sovereign_data as SD
    SD.SURVIVOR_MULT = float(cand["survivor_mult"])
    SD.KILL_MULT = float(cand["kill_mult"])
    SD.TAU = float(cand["tau"])
    SD.WIN_BONUS = float(cand["win_bonus"])
# EVOLVE-BLOCK-END


def proxy_score(cand, shards_dir, steps=12, seeds=2):
    """Cheap proxy: short PPO on farmer net + capped-lobby eval."""
    apply_weights(cand)
    import numpy as np
    import train_nn as T
    net = T.make_net()
    eps = []
    for p in sorted(Path(shards_dir).glob("*.npz"))[:6]:
        d = np.load(p, allow_pickle=True)
        eps += __import__("rl_loop", fromlist=["x"]).unpack_v2_episodes(d)
    if not eps:
        return -99.0
    opt = T.make_optimizer(net)
    T._train_ppo_batch(net, opt, eps, epochs=1)
    wr, rank = T.evaluate(net, seeds=seeds, silent=True)
    return wr * 10.0 - rank  # higher better


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--cands", default="candidates.json")
    ap.add_argument("--shards", default="weights/nn/rl/shards_v2")
    a = ap.parse_args()
    pop = json.load(open(a.cands)) if os.path.exists(a.cands) else DEFAULT_POP
    scored = []
    for c in pop:
        s = proxy_score(c, a.shards)
        scored.append((s, c))
        print(f"gen{a.gen} score={s:.3f} {c['note']}", flush=True)
    scored.sort(key=lambda x: -x[0])
    json.dump([{"score": s, **c} for s, c in scored],
              open(f"gen{a.gen}_scores.json", "w"), indent=1)
    print("TOP2 survive:", [c["note"] for _, c in scored[:2]], flush=True)


if __name__ == "__main__":
    main()
