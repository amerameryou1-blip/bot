#!/usr/bin/env python3
"""nano_rollout.py — B's answer to A's Q4: the env wrapper for SOVEREIGN-nano.

Runs game6 lobbies with nano.act() per tick, carrying a["state"] (GRU)
through each episode. Mirrors train_nn.evaluate's HONEST last-survivor
semantics ("win" = only player alive).

Usage:
  python3 scripts/nano_rollout.py [--ckpt path.pt] [--seeds N] [--max-ticks T]
No ckpt -> random-weight nano (smoke only).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch


def load_nano(ckpt=None):
    from nn.sovereign_nano import Nano, NanoConfig, make_nano
    net = make_nano()
    if ckpt and Path(ckpt).exists():
        blob = torch.load(ckpt, map_location="cpu")
        if isinstance(blob, dict) and "state_dict" in blob:
            cfg = blob.get("config")
            if cfg is not None:
                net = Nano(cfg)
            net.load_state_dict(blob["state_dict"])
        else:
            net.load_state_dict(blob)
    net.eval()
    return net


def nums_of(game, st):
    total = float(game.h * game.w)
    me_frac = st.self_blob.area / total if st.self_blob else 0.0
    enemy_frac = max((e.area for e in st.enemies), default=0) / total
    return torch.tensor([[0.0, me_frac, enemy_frac, 0.0, 0.0,
                          game.tick / max(game.max_ticks, 1), 0.0, 0.0]],
                        dtype=torch.float32)


def evaluate_nano(net, seeds=4, max_ticks=0, rtg_asp=0.0, silent=False):
    import train_nn as T
    from bot.planner import ClickAction
    wins = survived = 0
    ranks = []
    t0 = time.time()
    for seed in range(1, seeds + 1):
        game = T._make_game("mixed", seed, n_bots=T.SIM["n_bots"])
        if max_ticks:
            game.max_ticks = max_ticks
        state = None
        while game.tick < game.max_ticks:
            if not game.players[1].alive:
                break
            st = game.state_for(1)
            if not st.self_blob:
                break
            rgb, _ = game.frame_tensor(1, size=256)
            x = torch.tensor(rgb.transpose(2, 0, 1)[None],
                             dtype=torch.float32) / 255.0
            with torch.no_grad():
                a = net.act(x, nums_of(game, st),
                            rtg=torch.tensor([rtg_asp]), state=state)
            state = a["state"]          # GRU carry (A's proposal)
            kind_i = int(a["kind"][0])
            pctv = float(a["pct"][0])
            if kind_i == 2:
                act = ClickAction("bank", reason="nano-bank")
            else:
                cy = int(a["yx"][0][0]); cx = int(a["yx"][0][1])
                y = (cy + 0.5) / 256.0 * game.h
                xp = (cx + 0.5) / 256.0 * game.w
                kind_s = {0: "expand", 1: "attack"}[kind_i]
                act = ClickAction(kind_s, float(xp), float(y),
                                  pctv * 100.0 if kind_i == 1 else 0.0,
                                  reason=f"nano-{kind_s}")
            actions = {1: game._clicks_for(act, T.SIM["clicks_per_tick"])}
            for pid in game._pids:
                if pid == 1 or not game.players[pid].alive:
                    continue
                actions[pid] = game._bot_clicks(pid)
            game.step(actions)
        alive = game.players[1].alive and (game.world == 1).sum() > 0
        alive_list = [pid for pid in game._pids if game.players[pid].alive]
        is_last = bool(alive) and len(alive_list) == 1
        wins += 1 if is_last else 0
        survived += 1 if alive else 0
        areas = {pid: int((game.world == pid).sum()) for pid in game._pids}
        rank = 1 + sum(1 for pid, ar in areas.items()
                       if pid != 1 and ar > areas[1])
        ranks.append(rank)
    wr = wins / seeds
    avg_rank = sum(ranks) / len(ranks)
    if not silent:
        print(f"EVAL-NANO ({seeds} seeds): wr={wr:.2f} "
              f"alive={survived / seeds:.2f} rank={avg_rank:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return wr, avg_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max-ticks", type=int, default=0)
    a = ap.parse_args()
    net = load_nano(a.ckpt)
    evaluate_nano(net, seeds=a.seeds, max_ticks=a.max_ticks)


if __name__ == "__main__":
    main()
