#!/usr/bin/env python3
"""Train the neural bot: vision (seg + localization), behavior cloning of the
meta teacher, then PPO fine-tuning to maximize last-survivor wins.

Stages (each saves checkpoints so later stages can resume):
  1. collect  — play teacher matches -> dataset.npz
  2. vision   — segmentation + centroid (supervised)
  3. clone    — behavior clone the teacher's clicks (supervised)
  4. ppo      — policy-gradient fine-tune against the simulator (optional)

Run:  PYTHONPATH=src python3 scripts/train_nn.py [stage]
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from nn.model import TerritoryNet, NUM_CLASSES, count_params
from nn import data as ndata

WEIGHTS = REPO / "weights" / "nn"
WEIGHTS.mkdir(parents=True, exist_ok=True)
DATASET = WEIGHTS / "dataset.npz"
MODEL_PT = WEIGHTS / "model.pt"
CFG_JSON = WEIGHTS / "config.json"

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

GRID = 16
CTX_DIM = 3  # my_area_frac, enemy_area_frac, red_flag


def ctx_from_state(st, game, tracker_density=None):
    """Context vector: [my_area_frac, max_enemy_area_frac, red_flag]."""
    total = game.h * game.w
    my_frac = st.self_blob.area / total if st.self_blob else 0.0
    enemy_frac = max((e.area for e in st.enemies), default=0) / total
    red = 1.0 if (tracker_density is not None and tracker_density >= 90) else 0.0
    return np.array([my_frac, enemy_frac, red], dtype=np.float32)


def make_ctx_tensor(ctxs, device):
    return torch.tensor(np.stack(ctxs), device=device)


# ---------------------------------------------------------------- collect ---
def stage_collect(n=10000, seeds=16, per_seed=30):
    if DATASET.exists():
        print(f"dataset exists ({DATASET.stat().st_size/1e6:.1f} MB) — loading")
        return ndata.load(DATASET)
    print("collecting dataset (teacher matches, ε-greedy)...")
    data = ndata.collect(seeds=seeds, per_seed=per_seed, h=130, w=170, n_bots=3,
                         max_ticks=1400, eps=0.15, clicks_per_tick=12)
    # trim to n
    if len(data["rgb"]) > n:
        idx = np.random.default_rng(0).choice(len(data["rgb"]), n, replace=False)
        data = {k: v[idx] for k, v in data.items()}
    ndata.save(data, DATASET)
    return data


# ---------------------------------------------------------------- vision ----
def stage_vision(net, data, epochs=6, bs=256, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    seg_w = torch.tensor([1.0, 1.2, 3.0, 2.0, 1.0])  # weight me/enemy higher
    X = torch.tensor(data["rgb"]); Y = torch.tensor(data["seg"])
    C = torch.tensor(data["centroid"])
    n = len(X)
    print(f"vision training: {n} samples, {count_params(net)} params")
    for ep in range(epochs):
        net.train()
        tot = 0.0; nb = 0
        for i in range(0, n, bs):
            xb, yb, cb = X[i:i+bs], Y[i:i+bs], C[i:i+bs]
            seg, local, *_ = net.forward(xb, return_all=True)
            # seg: (B,5,16,16); labels downsampled to 16x16
            yb16 = F.interpolate(yb.float().unsqueeze(1), size=(GRID, GRID), mode="nearest").long().squeeze(1)
            loss = F.cross_entropy(seg, yb16, weight=seg_w) + F.mse_loss(local, cb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  vision epoch {ep+1}: loss={tot/nb:.4f}")
    return net


# ---------------------------------------------------------------- clone -----
def stage_clone(net, data, epochs=5, bs=256, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    X = torch.tensor(data["rgb"])
    K = torch.tensor(data["kind"]); CEL = torch.tensor(data["cell"]); P = torch.tensor(data["pct"])
    n = len(X)
    print(f"clone training: {n} samples")
    for ep in range(epochs):
        net.train()
        tot = 0.0; nb = 0
        for i in range(0, n, bs):
            xb, kb, cb, pb = X[i:i+bs], K[i:i+bs], CEL[i:i+bs], P[i:i+bs]
            ctx = torch.zeros(xb.shape[0], CTX_DIM)
            click, kind, pct, _ = net.forward(xb, ctx)
            loss = F.cross_entropy(kind, kb)
            # cell loss only on non-bank samples
            mask = kb != 2
            if mask.any():
                loss = loss + F.cross_entropy(click[mask], cb[mask])
            # pct loss on attack samples
            am = kb == 1
            if am.any():
                loss = loss + F.mse_loss(pct[am], pb[am])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  clone epoch {ep+1}: loss={tot/nb:.4f}")
    return net


# ---------------------------------------------------------------- eval ------
def evaluate(net, seeds=6, n_bots=3, greedy=True):
    """Win rate (last survivor) + rank using the net as the policy in game5."""
    from sim.game5 import ClickSim5
    from bot.planner import ClickPlanner, ClickPlannerConfig
    from bot.economy import TroopTracker
    import torch

    def policy_fn(state, game, planner_tracker):
        # build obs from the sim state directly (no rendering needed for eval
        # of the policy — but we want to test the REAL pipeline: render pixels)
        rgb, _ = game.frame_tensor(1, size=64)
        x = torch.tensor(rgb.transpose(2, 0, 1)[None], dtype=torch.float32) / 255.0
        ctx = torch.tensor(ctx_from_state(state, game, planner_tracker.density())[None])
        net.eval()
        with torch.no_grad():
            seg, local, click, kind, pct, _ = net.forward(x, ctx, return_all=True)
            kind_i = int(kind[0].argmax())
            if kind_i == 2:
                from bot.planner import ClickAction
                return ClickAction("bank", reason="nn-bank")
            cell = int(click[0].argmax()) if greedy else int(torch.multinomial(F.softmax(click[0], dim=-1), 1))
            cy, cx = divmod(cell, GRID)
            y = (cy + 0.5) / GRID * game.h
            xp = (cx + 0.5) / GRID * game.w
            from bot.planner import ClickAction
            kind_s = {0: "expand", 1: "attack"}[kind_i]
            pct_v = float(pct[0]) * 100.0 if kind_i == 1 else 0.0
            return ClickAction(kind_s, float(xp), float(y), pct_v, reason=f"nn-{kind_s}")

    wins = 0; ranks = []
    for seed in range(1, seeds + 1):
        game = ClickSim5(h=130, w=170, n_bots=n_bots, seed=seed, max_ticks=1600, clicks_per_tick=12)
        planner_tracker = TroopTracker(balance=512.0, land=12)
        while game.tick < game.max_ticks:
            actions = {}
            if game.players[1].alive:
                st = game.state_for(1)
                if st.self_blob:
                    planner_tracker.update(st.self_blob.area)
                    act = policy_fn(st, game, planner_tracker)
                    actions[1] = game._clicks_for(act, 12)
            for pid in game._pids:
                if pid == 1 or not game.players[pid].alive:
                    continue
                actions[pid] = game._bot_clicks(pid)
            game.step(actions)
        wins += 1 if game.players[1].alive and (game.world == 1).sum() > 0 else 0
        areas = {pid: int((game.world == pid).sum()) for pid in game._pids}
        rank = 1 + sum(1 for pid, a in areas.items() if pid != 1 and a > areas[1])
        ranks.append(rank)
    wr = wins / seeds
    print(f"EVAL: alive-rate={wr:.2f} avg_rank={sum(ranks)/len(ranks):.2f} (seeds={seeds})")
    return wr


# ---------------------------------------------------------------- PPO -------
def stage_ppo(net, episodes=80, lr=1e-4, gamma=0.99, lam=0.95, clip=0.2,
              epochs_per_batch=4, n_bots=3, max_ticks=1600):
    """Compact PPO: play episodes in game5, collect (obs, action, reward),
    update with clipped surrogate + GAE. Reward: +1 win, -1 death, small area
    shaping, +kill bonus."""
    from sim.game5 import ClickSim5
    from bot.planner import ClickAction

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    log_steps = 0
    for ep in range(1, episodes + 1):
        game = ClickSim5(h=130, w=170, n_bots=n_bots, seed=1000 + ep, max_ticks=max_ticks, clicks_per_tick=12)
        ob_rgb, ob_ctx, acts_k, acts_c, acts_p, logp, rews = [], [], [], [], [], [], []
        done = False
        while game.tick < game.max_ticks:
            if not game.players[1].alive:
                break
            st = game.state_for(1)
            if not st.self_blob:
                break
            rgb, _ = game.frame_tensor(1, size=64)
            x = torch.tensor(rgb.transpose(2, 0, 1)[None], dtype=torch.float32) / 255.0
            ctx_t = torch.tensor(ctx_from_state(st, game)[None])
            with torch.no_grad():
                click, kind, pct, value = net.forward(x, ctx_t)
                probs_k = F.softmax(kind[0], dim=-1)
                probs_c = F.softmax(click[0], dim=-1)
                kind_i = int(torch.multinomial(probs_k, 1))
                if kind_i == 2:
                    cell_i = 0
                else:
                    cell_i = int(torch.multinomial(probs_c, 1))
                pct_v = float(pct[0])
            # execute
            cy, cx = divmod(cell_i, GRID)
            y = (cy + 0.5) / GRID * game.h
            xp = (cx + 0.5) / GRID * game.w
            kind_s = {0: "expand", 1: "attack", 2: "bank"}[kind_i]
            act = ClickAction(kind_s, float(xp), float(y), pct_v * 100.0 if kind_i == 1 else 0.0,
                              reason=f"ppo-{kind_s}")
            actions = {1: game._clicks_for(act, 12)}
            for pid in game._pids:
                if pid == 1 or not game.players[pid].alive:
                    continue
                actions[pid] = game._bot_clicks(pid)
            area_before = int((game.world == 1).sum())
            game.step(actions)
            area_after = int((game.world == 1).sum())
            # reward shaping
            reward = (area_after - area_before) / 5000.0  # small growth signal
            if not game.players[1].alive:
                reward -= 1.0
                done = True
            ob_rgb.append(x[0]); ob_ctx.append(ctx_t[0])
            acts_k.append(kind_i); acts_c.append(cell_i); acts_p.append(pct_v)
            with torch.no_grad():
                lp = torch.log(probs_k[kind_i]) + (torch.log(probs_c[cell_i]) if kind_i != 2 else 0.0)
            logp.append(lp)
            rews.append(reward)
            if done:
                break
        # end-of-episode: win bonus / alive bonus
        alive_final = game.players[1].alive
        final_area = int((game.world == 1).sum())
        rews[-1] += (1.0 if alive_final else 0.0)  # survival bonus
        rews[-1] += min(final_area / 20000.0, 1.0) * 0.2

        # GAE
        T = len(rews)
        if T < 3:
            continue
        with torch.no_grad():
            vals = []
            for i in range(T):
                rgb_i = ob_rgb[i].unsqueeze(0); ctx_i = ob_ctx[i].unsqueeze(0)
                _, _, _, v = net.forward(rgb_i, ctx_i)
                vals.append(float(v[0]))
            adv = np.zeros(T); gae = 0.0
            for t in reversed(range(T)):
                nxt = vals[t + 1] if t + 1 < T else (1.0 if alive_final else 0.0)
                delta = rews[t] + gamma * nxt - vals[t]
                gae = delta + gamma * lam * gae
                adv[t] = gae
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # one update pass over this episode
        Xb = torch.stack(ob_rgb); Cb = torch.stack(ob_ctx)
        Kb = torch.tensor(acts_k); CELb = torch.tensor(acts_c); Pb = torch.tensor(acts_p)
        old_logp = torch.stack(logp)
        Adv = torch.tensor(adv, dtype=torch.float32)
        Ret = torch.tensor([rews[t] + (vals[t + 1] if t + 1 < T else (1.0 if alive_final else 0.0)) * gamma
                            for t in range(T)], dtype=torch.float32)
        for _ in range(epochs_per_batch):
            click, kind, pct, value = net.forward(Xb, Cb)
            probs_k = F.log_softmax(kind, dim=1); probs_c = F.log_softmax(click, dim=1)
            lp = probs_k.gather(1, Kb.unsqueeze(1)).squeeze(1)
            nonbank = Kb != 2
            if nonbank.any():
                lp = lp + probs_c.gather(1, CELb[nonbank].unsqueeze(1)).squeeze(1) * nonbank.float()
            ratio = torch.exp(lp - old_logp)
            pg = -torch.min(ratio * Adv, torch.clamp(ratio, 1 - clip, 1 + clip) * Adv).mean()
            vf = F.mse_loss(value, Ret)
            ent = -torch.mean(probs_k * probs_k.exp()) - (torch.mean(probs_c * probs_c.exp()) if nonbank.any() else 0.0)
            loss = pg + 0.5 * vf - 0.01 * ent
            opt.zero_grad(); loss.backward(); opt.step()
        log_steps += T
        if ep % 10 == 0:
            print(f"  ppo ep {ep}: steps={log_steps} win={alive_final} final_area={final_area}")
    return net


# ---------------------------------------------------------------- main ------
def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    device = "cpu"
    net = TerritoryNet(grid=GRID, context_dim=CTX_DIM)
    if MODEL_PT.exists():
        net.load_state_dict(torch.load(MODEL_PT, map_location=device))
        print("loaded existing checkpoint")
    if stage in ("collect", "all"):
        data = stage_collect()
    else:
        data = ndata.load(DATASET) if DATASET.exists() else stage_collect()
    if stage in ("vision", "all"):
        net = stage_vision(net, data)
        torch.save(net.state_dict(), MODEL_PT)
    if stage in ("clone", "all"):
        net = stage_clone(net, data)
        torch.save(net.state_dict(), MODEL_PT)
    if stage in ("ppo", "all"):
        net = stage_ppo(net, episodes=int(sys.argv[2]) if len(sys.argv) > 2 else 80)
        torch.save(net.state_dict(), MODEL_PT)
    # final eval + export
    evaluate(net, seeds=6)
    json.dump({"grid": GRID, "context_dim": CTX_DIM, "classes": NUM_CLASSES},
              open(CFG_JSON, "w"), indent=2)
    torch.save(net.state_dict(), MODEL_PT)
    print(f"saved model -> {MODEL_PT}")
    print(f"params: {count_params(net)}")


if __name__ == "__main__":
    main()
