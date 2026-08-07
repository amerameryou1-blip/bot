"""Dataset collection — parallel, skill-curriculum aware.

Each worker plays matches in the simulator (teacher or noisy policy) and
records (screen RGB, segmentation labels, centroid, action). Runs in multiple
processes so collection keeps the GPU fed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from bot.planner import ClickPlanner, ClickPlannerConfig
from bot.economy import TroopTracker
from sim.game5 import ClickSim5

KIND_TO_IDX = {"expand": 0, "attack": 1, "bank": 2}
CELLS = 16
RECORD_EVERY = 4  # record 1 in 4 ticks (frames are near-duplicates)


def _to_cell(y, x, h, w):
    cy = min(CELLS - 1, int(y / h * CELLS))
    cx = min(CELLS - 1, int(x / w * CELLS))
    return cy * CELLS + cx


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


def collect_seed(sd, n_bots, h, w, max_ticks, clicks_per_tick, eps, bot_skill, rng_seed, record_every):
    """Play `n_matches` matches for one seed; return sample arrays."""
    rng = np.random.default_rng(rng_seed)
    rgb_l, seg_l, cent_l, kind_l, cell_l, pct_l = [], [], [], [], [], []
    planner = ClickPlanner(ClickPlannerConfig(), TroopTracker(balance=512.0, land=12))
    game = ClickSim5(h=h, w=w, n_bots=n_bots, seed=sd, max_ticks=max_ticks,
                     clicks_per_tick=clicks_per_tick, bot_skill=bot_skill)
    while game.tick < game.max_ticks:
        actions = {}
        if game.players[1].alive:
            st = game.state_for(1)
            if st.self_blob:
                planner.set_enemy_balances(
                    {f"e{pid}": game.players[pid].troops.balance for pid in game._pids if pid != 1})
                act = planner.decide(st)
                if game.tick % record_every == 0:
                    rgb, seg = game.frame_tensor(1, size=64)
                    me = st.self_blob
                    rgb_l.append(rgb.transpose(2, 0, 1))
                    seg_l.append(seg)
                    cent_l.append([me.centroid[0] / game.h, me.centroid[1] / game.w])
                    kind_l.append(KIND_TO_IDX[act.kind])
                    cell_l.append(_to_cell(act.y, act.x, game.h, game.w) if act.kind != "bank" else 0)
                    pct_l.append(act.pct / 100.0 if act.kind == "attack" else 0.0)
                if rng.random() < eps:
                    act = _random_action(st, rng, game)
                actions[1] = game._clicks_for(act, clicks_per_tick)
        for pid in game._pids:
            if pid == 1 or not game.players[pid].alive:
                continue
            actions[pid] = game._bot_clicks(pid)
        game.step(actions)
    if not rgb_l:
        return None
    return {
        "rgb": np.array(rgb_l, dtype=np.float32) / 255.0,
        "seg": np.array(seg_l, dtype=np.int64),
        "centroid": np.array(cent_l, dtype=np.float32),
        "kind": np.array(kind_l, dtype=np.int64),
        "cell": np.array(cell_l, dtype=np.int64),
        "pct": np.array(pct_l, dtype=np.float32),
    }


def _worker(args):
    return collect_seed(*args)


def collect_parallel(seeds=24, n_bots=3, h=110, w=140, max_ticks=1400, clicks_per_tick=12,
                     eps=0.15, bot_skill="medium", record_every=RECORD_EVERY, workers=4):
    """Collect from many seeds in parallel; combine into one array dict."""
    import multiprocessing as mp
    tasks = [(sd, n_bots, h, w, max_ticks, clicks_per_tick, eps, bot_skill, sd * 7 + 1, record_every)
             for sd in range(1, seeds + 1)]
    results = []
    with mp.Pool(workers) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            if res is not None:
                results.append(res)
    if not results:
        return None
    keys = ["rgb", "seg", "centroid", "kind", "cell", "pct"]
    out = {k: np.concatenate([r[k] for r in results]) for k in keys}
    return out


def collect_single(seeds=4, **kw):
    """Fallback (no multiprocessing) — collect sequentially."""
    results = []
    for sd in range(1, seeds + 1):
        r = collect_seed(sd, kw.get("n_bots", 3), kw.get("h", 110), kw.get("w", 140),
                         kw.get("max_ticks", 1400), kw.get("clicks_per_tick", 12),
                         kw.get("eps", 0.15), kw.get("bot_skill", "medium"), sd * 7 + 1,
                         kw.get("record_every", RECORD_EVERY))
        if r is not None:
            results.append(r)
    if not results:
        return None
    keys = ["rgb", "seg", "centroid", "kind", "cell", "pct"]
    return {k: np.concatenate([r[k] for r in results]) for k in keys}


def save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"saved {len(data['rgb'])} samples -> {path}")


def load(path):
    d = np.load(path)
    return {k: d[k] for k in ("rgb", "seg", "centroid", "kind", "cell", "pct")}
