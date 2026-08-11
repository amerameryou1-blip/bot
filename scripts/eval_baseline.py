#!/usr/bin/env python3
"""Honest evidence run: current net vs RANDOM policy, identical eval protocol
(last-survivor wins only, mixed 10-bot lobbies, seeds 1..N). Answers
'learned or random?' with a controlled comparison."""
import os
import sys
import time
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import train_nn as T
from bot.planner import ClickAction


def eval_policy(policy_fn, seeds=6, tag=""):
    wins = survived = 0
    ranks = []
    for seed in range(1, seeds + 1):
        game = T._make_game("mixed", seed, n_bots=T.SIM["n_bots"])
        while game.tick < game.max_ticks:
            if not game.players[1].alive:
                break
            st = game.state_for(1)
            if not st.self_blob:
                break
            act, _ = policy_fn(st, game)
            actions = {1: game._clicks_for(act, T.SIM["clicks_per_tick"])}
            for pid in game._pids:
                if pid == 1 or not game.players[pid].alive:
                    continue
                actions[pid] = game._bot_clicks(pid)
            game.step(actions)
        alive = game.players[1].alive and (game.world == 1).sum() > 0
        alive_list = [p for p in game._pids if game.players[p].alive]
        wins += 1 if (alive and len(alive_list) == 1) else 0
        survived += 1 if alive else 0
        areas = {p: int((game.world == p).sum()) for p in game._pids}
        ranks.append(1 + sum(1 for p, a in areas.items() if p != 1 and a > areas[1]))
    wr = wins / seeds
    print(f"{tag}: last-survivor WR={wr:.2f} alive={survived/seeds:.2f} "
          f"avg_rank={np.mean(ranks):.2f}", flush=True)
    return wr


def main():
    seeds = int(os.environ.get("EVAL_SEEDS", "6"))
    net = T.make_net().to(T.DEVICE)
    T.load_model(net)
    net.eval()

    def net_policy(st, game):
        return T._policy_action(net, st, game)

    rng = np.random.RandomState(0)

    def rand_policy(st, game):
        r = rng.rand()
        if r < 0.1:
            return ClickAction("bank", reason="rand"), 0.5
        kind = "expand" if r < 0.55 else "attack"
        x = float(rng.randint(0, game.w)); y = float(rng.randint(0, game.h))
        pct = float(rng.randint(5, 100)) if kind == "attack" else 0.0
        return ClickAction(kind, x, y, pct, reason="rand"), 0.5

    t0 = time.time()
    wr_net = eval_policy(net_policy, seeds, "NET ")
    wr_rand = eval_policy(rand_policy, seeds, "RAND")
    print(f"delta WR (net - random) = {wr_net - wr_rand:+.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
