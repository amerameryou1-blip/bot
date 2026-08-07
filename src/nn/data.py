"""Dataset collection: play matches in game5 with the meta teacher (ε-greedy)
and record (screen RGB, segmentation labels, my centroid, teacher action).

Each sample:
  rgb     : (3, 64, 64) float 0..1
  seg     : (64, 64) long     class ids (0 water,1 neutral,2 me,3 enemy)
  centroid: (2,) float       my normalized position
  kind    : int              0 expand, 1 attack, 2 bank
  cell    : int              16x16 cell index the teacher clicked (0 if bank)
  pct     : float            0..1 attack slider (0 if not attack)
  reward  : float            (filled by RL; 0 here for cloning)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sim.game5 import ClickSim5
from bot.planner import ClickPlanner, ClickPlannerConfig
from bot.economy import TroopTracker

KIND_TO_IDX = {"expand": 0, "attack": 1, "bank": 2}
IDX_TO_KIND = {0: "expand", 1: "attack", 2: "bank"}

CELLS = 16


def _to_cell(y, x, h, w):
    cy = min(CELLS - 1, int(y / h * CELLS))
    cx = min(CELLS - 1, int(x / w * CELLS))
    return cy * CELLS + cx


def collect(seeds=12, per_seed=40, h=130, w=170, n_bots=3, max_ticks=1400,
            eps=0.15, clicks_per_tick=12, seed_rng=0, out_dir=None):
    """Run teacher matches with ε-greedy noise; return dict of numpy arrays."""
    rng = np.random.default_rng(seed_rng)
    rgb_list, seg_list, cent_list = [], [], []
    kind_list, cell_list, pct_list = [], [], []

    def teacher_for_seed(sd):
        cfg = ClickPlannerConfig()
        return ClickPlanner(cfg, TroopTracker(balance=512.0, land=12))

    for sd in range(1, seeds + 1):
        planner = teacher_for_seed(sd)
        for run in range(per_seed):
            game = ClickSim5(h=h, w=w, n_bots=n_bots, seed=sd * 100 + run,
                             max_ticks=max_ticks, clicks_per_tick=clicks_per_tick)
            while game.tick < game.max_ticks:
                actions = {}
                if game.players[1].alive:
                    st = game.state_for(1)
                    if st.self_blob:
                        planner.set_enemy_balances(
                            {f"e{pid}": game.players[pid].troops.balance for pid in game._pids if pid != 1})
                        act = planner.decide(st)
                        # record BEFORE executing (teacher's choice for this state)
                        rgb, seg = game.frame_tensor(1, size=64)
                        me = st.self_blob
                        cy_n = me.centroid[0] / game.h
                        cx_n = me.centroid[1] / game.w
                        kind_i = KIND_TO_IDX[act.kind]
                        cell_i = _to_cell(act.y, act.x, game.h, game.w) if act.kind != "bank" else 0
                        pct = act.pct / 100.0 if act.kind == "attack" else 0.0
                        rgb_list.append(rgb.transpose(2, 0, 1))
                        seg_list.append(seg)
                        cent_list.append([cy_n, cx_n])
                        kind_list.append(kind_i)
                        cell_list.append(cell_i)
                        pct_list.append(pct)

                        # ε-greedy: occasionally deviate for coverage
                        if rng.random() < eps:
                            act = _random_action(st, rng, game)
                        actions[1] = game._clicks_for(act, clicks_per_tick)
                for pid in game._pids:
                    if pid == 1 or not game.players[pid].alive:
                        continue
                    actions[pid] = game._bot_clicks(pid)
                game.step(actions)
                if len(rgb_list) >= seeds * per_seed * 8:
                    break
            if len(rgb_list) >= seeds * per_seed * 8:
                break
        if len(rgb_list) >= seeds * per_seed * 8:
            break

    data = {
        "rgb": np.array(rgb_list, dtype=np.float32) / 255.0,
        "seg": np.array(seg_list, dtype=np.int64),
        "centroid": np.array(cent_list, dtype=np.float32),
        "kind": np.array(kind_list, dtype=np.int64),
        "cell": np.array(cell_list, dtype=np.int64),
        "pct": np.array(pct_list, dtype=np.float32),
    }
    return data


def _random_action(st, rng, game):
    from bot.planner import ClickAction
    r = rng.random()
    if r < 0.4 and len(st.expand_targets):
        t = st.expand_targets[rng.integers(len(st.expand_targets))]
        return ClickAction("expand", float(t[1]), float(t[0]), reason="rand-expand")
    if r < 0.8 and len(st.attack_targets):
        t = st.attack_targets[rng.integers(len(st.attack_targets))]
        return ClickAction("attack", float(t[1]), float(t[0]), 5.0 + rng.random() * 15.0, reason="rand-attack")
    return ClickAction("bank", reason="rand-bank")


def save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    n = len(data["rgb"])
    print(f"saved {n} samples -> {path}")


def load(path):
    d = np.load(path)
    return {k: d[k] for k in ("rgb", "seg", "centroid", "kind", "cell", "pct")}


def to_tensors(data, device="cpu"):
    return {
        "rgb": torch.tensor(data["rgb"], device=device),
        "seg": torch.tensor(data["seg"], device=device),
        "centroid": torch.tensor(data["centroid"], device=device),
        "kind": torch.tensor(data["kind"], device=device),
        "cell": torch.tensor(data["cell"], device=device),
        "pct": torch.tensor(data["pct"], device=device),
    }
