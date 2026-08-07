#!/usr/bin/env python3
"""Neural bot trainer — the FULL pipeline, GPU-maximized.

Stages:
  collect — parallel simulated matches (teacher, ε-greedy) -> dataset.npz
  vision  — train segmentation + localization (supervised, GPU)
  clone   — behavior-clone the teacher's clicks (supervised, GPU)
  ppo     — growing-replay PPO vs the simulator, difficulty curriculum
            (medium -> hard by win rate), parallel rollouts on CPU while
            the GPU trains
  export  — push weights to Hugging Face (HF_TOKEN)

GPU is the trainer; CPU cores run the simulator. One session does it all.

Run: PYTHONPATH=src python3 scripts/train_nn.py [stage] [extra]
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
from sim.game5 import ClickSim5
from bot.planner import ClickAction

WEIGHTS = REPO / "weights" / "nn"
WEIGHTS.mkdir(parents=True, exist_ok=True)
DATASET = WEIGHTS / "dataset.npz"
MODEL_PT = WEIGHTS / "model.pt"
CFG_JSON = WEIGHTS / "config.json"

GRID = 16
CTX_DIM = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}", flush=True)

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

# sim params for training
SIM = dict(h=110, w=140, n_bots=3, max_ticks=1400, clicks_per_tick=12)


def ctx_from_state(st, game, density=None):
    total = game.h * game.w
    my_frac = st.self_blob.area / total if st.self_blob else 0.0
    enemy_frac = max((e.area for e in st.enemies), default=0) / total
    red = 1.0 if (density is not None and density >= 90) else 0.0
    return np.array([my_frac, enemy_frac, red], dtype=np.float32)


def make_net():
    return TerritoryNet(grid=GRID, context_dim=CTX_DIM)


def save_model(net):
    torch.save(net.state_dict(), MODEL_PT)
    json.dump({"grid": GRID, "context_dim": CTX_DIM, "classes": NUM_CLASSES},
              open(CFG_JSON, "w"), indent=2)


def load_model(net):
    if MODEL_PT.exists():
        net.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
        print("loaded checkpoint", flush=True)


# ================================================================ collect ====
def stage_collect(seeds=28, bot_skill="medium", workers=4):
    if DATASET.exists():
        print("dataset exists — loading", flush=True)
        return
    print(f"collecting (skill={bot_skill}, seeds={seeds}, workers={workers})...", flush=True)
    t0 = time.time()
    try:
        data = ndata.collect_parallel(seeds=seeds, n_bots=SIM["n_bots"], h=SIM["h"], w=SIM["w"],
                                      max_ticks=SIM["max_ticks"], clicks_per_tick=SIM["clicks_per_tick"],
                                      eps=0.15, bot_skill=bot_skill, workers=workers)
    except Exception as e:
        print("parallel collect failed, falling back to single:", e, flush=True)
        data = ndata.collect_single(seeds=seeds, n_bots=SIM["n_bots"], h=SIM["h"], w=SIM["w"],
                                    max_ticks=SIM["max_ticks"], clicks_per_tick=SIM["clicks_per_tick"],
                                    eps=0.15, bot_skill=bot_skill)
    if data is None:
        raise RuntimeError("no data collected")
    ndata.save(data, DATASET)
    print(f"collected in {time.time()-t0:.0f}s", flush=True)


# ================================================================ vision ====
def stage_vision(net, epochs=8, bs=256, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    seg_w = torch.tensor([1.0, 1.2, 3.0, 2.0, 1.0], device=DEVICE)
    d = np.load(DATASET, mmap_mode="r")
    n = len(d["rgb"])
    print(f"vision: {n} samples, {count_params(net)} params, device={DEVICE}", flush=True)
    for ep in range(epochs):
        net.train()
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            xb = torch.tensor(d["rgb"][i:i+bs], dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(d["seg"][i:i+bs], dtype=torch.int64, device=DEVICE)
            cb = torch.tensor(d["centroid"][i:i+bs], dtype=torch.float32, device=DEVICE)
            seg, local, *_ = net.forward(xb, return_all=True)
            yb16 = F.interpolate(yb.float().unsqueeze(1), size=(GRID, GRID), mode="nearest").long().squeeze(1)
            loss = F.cross_entropy(seg, yb16, weight=seg_w) + F.mse_loss(local, cb) * 5.0
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  vision epoch {ep+1}: loss={tot/nb:.4f}", flush=True)
    return net


# ================================================================ clone ======
def stage_clone(net, epochs=8, bs=256, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    d = np.load(DATASET, mmap_mode="r")
    n = len(d["rgb"])
    kind_w = torch.tensor([1.0, 4.0, 1.5], device=DEVICE)
    print(f"clone: {n} samples", flush=True)
    for ep in range(epochs):
        net.train()
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            xb = torch.tensor(d["rgb"][i:i+bs], dtype=torch.float32, device=DEVICE)
            kb = torch.tensor(d["kind"][i:i+bs], dtype=torch.int64, device=DEVICE)
            cb = torch.tensor(d["cell"][i:i+bs], dtype=torch.int64, device=DEVICE)
            pb = torch.tensor(d["pct"][i:i+bs], dtype=torch.float32, device=DEVICE)
            ctx = torch.zeros(xb.shape[0], CTX_DIM, device=DEVICE)
            click, kind, pct, _ = net.forward(xb, ctx)
            loss = F.cross_entropy(kind, kb, weight=kind_w)
            mask = kb != 2
            if mask.any():
                loss = loss + F.cross_entropy(click[mask], cb[mask]) * 1.5
            am = kb == 1
            if am.any():
                loss = loss + F.mse_loss(pct[am], pb[am]) * 2.0
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  clone epoch {ep+1}: loss={tot/nb:.4f}", flush=True)
    return net


# ================================================================ PPO ========
NET_G = None  # worker-global net


def _init_worker(state_dict):
    global NET_G
    NET_G = make_net()
    NET_G.load_state_dict(state_dict)
    NET_G.eval()


def _policy_action(net, st, game):
    rgb, _ = game.frame_tensor(1, size=64)
    x = torch.tensor(rgb.transpose(2, 0, 1)[None], dtype=torch.float32)
    with torch.no_grad():
        click, kind, pct, _ = net.forward(x, None)
        probs_k = F.softmax(kind[0], dim=-1)
        probs_c = F.softmax(click[0], dim=-1)
        kind_i = int(torch.multinomial(probs_k, 1))
        if kind_i == 2:
            return ClickAction("bank", reason="ppo-bank"), float(pct[0])
        cell = int(torch.multinomial(probs_c, 1))
        cy, cx = divmod(cell, GRID)
        y = (cy + 0.5) / GRID * game.h
        xp = (cx + 0.5) / GRID * game.w
        kind_s = {0: "expand", 1: "attack"}[kind_i]
        return ClickAction(kind_s, float(xp), float(y), float(pct[0]) * 100.0 if kind_i == 1 else 0.0,
                           reason=f"ppo-{kind_s}"), float(pct[0])


def _rollout_one(net, seed, skill, episodes=1):
    """Play `episodes` matches; return a list of episode dicts."""
    eps = []
    for e in range(episodes):
        game = ClickSim5(h=SIM["h"], w=SIM["w"], n_bots=SIM["n_bots"], seed=seed * 100 + e,
                         max_ticks=SIM["max_ticks"], clicks_per_tick=SIM["clicks_per_tick"],
                         bot_skill=skill)
        ob_rgb, ob_ctx, k, c, p, lp, rw = [], [], [], [], [], [], []
        done = False
        while game.tick < game.max_ticks:
            if not game.players[1].alive:
                break
            st = game.state_for(1)
            if not st.self_blob:
                break
            act, pctv = _policy_action(net, st, game)
            rgb, _ = game.frame_tensor(1, size=64)
            x = torch.tensor(rgb.transpose(2, 0, 1)[None], dtype=torch.float32)
            with torch.no_grad():
                click, kind, pct, _ = net.forward(x, None)
                probs_k = F.softmax(kind[0], dim=-1)
                probs_c = F.softmax(click[0], dim=-1)
                kind_i = int(torch.multinomial(probs_k, 1))
                cell_i = int(torch.multinomial(probs_c, 1)) if kind_i != 2 else 0
                lp_k = float(torch.log(probs_k[kind_i]))
                lp_c = float(torch.log(probs_c[cell_i])) if kind_i != 2 else 0.0
            ob_rgb.append(rgb.transpose(2, 0, 1))
            ob_ctx.append(ctx_from_state(st, game))
            k.append(kind_i); c.append(cell_i); p.append(pctv); lp.append(lp_k + lp_c)
            area_before = int((game.world == 1).sum())
            actions = {1: game._clicks_for(act, SIM["clicks_per_tick"])}
            for pid in game._pids:
                if pid == 1 or not game.players[pid].alive:
                    continue
                actions[pid] = game._bot_clicks(pid)
            game.step(actions)
            area_after = int((game.world == 1).sum())
            reward = (area_after - area_before) / 5000.0
            if not game.players[1].alive:
                reward -= 1.0
                done = True
            rw.append(reward)
            if done:
                break
        if len(rw) < 3:
            continue
        alive_final = game.players[1].alive
        final_area = int((game.world == 1).sum())
        rw[-1] += (1.0 if alive_final else 0.0) + min(final_area / 20000.0, 1.0) * 0.2
        eps.append({
            "rgb": np.array(ob_rgb, dtype=np.float32) / 255.0,
            "ctx": np.array(ob_ctx, dtype=np.float32),
            "kind": np.array(k, dtype=np.int64),
            "cell": np.array(c, dtype=np.int64),
            "pct": np.array(p, dtype=np.float32),
            "logp": np.array(lp, dtype=np.float32),
            "reward": np.array(rw, dtype=np.float32),
            "alive": bool(alive_final),
        })
    return eps


def _rollout_worker(args):
    seed, skill, episodes = args
    return _rollout_one(NET_G, seed, skill, episodes)


def _train_ppo_batch(net, opt, episodes, epochs=4, gamma=0.99, lam=0.95, clip=0.2):
    net.train()
    tot_loss, nb = 0.0, 0
    for ep in episodes:
        T = len(ep["reward"])
        # GAE
        with torch.no_grad():
            xb = torch.tensor(ep["rgb"], device=DEVICE)
            cxb = torch.tensor(ep["ctx"], device=DEVICE)
            _, _, _, vals = net.forward(xb, cxb)
            vals = vals.cpu().numpy()
        adv = np.zeros(T); gae = 0.0
        for t in reversed(range(T)):
            nxt = vals[t + 1] if t + 1 < T else (1.0 if ep["alive"] else 0.0)
            delta = ep["reward"][t] + gamma * nxt - vals[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = np.array([ep["reward"][t] + (vals[t + 1] if t + 1 < T else (1.0 if ep["alive"] else 0.0)) * gamma
                        for t in range(T)], dtype=np.float32)
        xb = torch.tensor(ep["rgb"], device=DEVICE)
        cxb = torch.tensor(ep["ctx"], device=DEVICE)
        kb = torch.tensor(ep["kind"], device=DEVICE)
        cb = torch.tensor(ep["cell"], device=DEVICE)
        pb = torch.tensor(ep["pct"], device=DEVICE)
        old_lp = torch.tensor(ep["logp"], device=DEVICE)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
        ret_t = torch.tensor(ret, device=DEVICE)
        for _ in range(epochs):
            click, kind, pct, value = net.forward(xb, cxb)
            logpk = F.log_softmax(kind, dim=1)
            logpc = F.log_softmax(click, dim=1)
            lp = logpk.gather(1, kb.unsqueeze(1)).squeeze(1)
            nb_mask = kb != 2
            if nb_mask.any():
                lp = lp + logpc.gather(1, cb[nb_mask].unsqueeze(1)).squeeze(1) * nb_mask.float()
            ratio = torch.exp(lp - old_lp)
            pg = -torch.min(ratio * adv_t, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t).mean()
            vf = F.mse_loss(value, ret_t)
            ent = -torch.mean(logpk * logpk.exp())
            loss = pg + 0.5 * vf - 0.01 * ent
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item(); nb += 1
    return tot_loss / max(nb, 1)


def stage_ppo(net, rounds=40, episodes_per_worker=2, workers=2, lr=1e-4, eval_every=5,
              pool_cap=6000):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    import multiprocessing as mp
    pool_eps = []
    skill = "medium"
    best_wr = 0.0
    for rnd in range(1, rounds + 1):
        net.eval()
        state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
        t0 = time.time()
        with mp.Pool(workers, initializer=_init_worker, initargs=(state_dict,)) as pool:
            tasks = [(1000 + rnd * 10 + w, skill, episodes_per_worker) for w in range(workers)]
            for res in pool.imap_unordered(_rollout_worker, tasks):
                pool_eps.extend(res)
        collect_t = time.time() - t0
        # cap pool (drop oldest)
        if len(pool_eps) > pool_cap:
            pool_eps = pool_eps[-pool_cap:]
        t0 = time.time()
        loss = _train_ppo_batch(net, opt, pool_eps)
        train_t = time.time() - t0
        alive_rate = sum(e["alive"] for e in pool_eps[-workers * episodes_per_worker:]) / max(workers * episodes_per_worker, 1)
        if rnd % eval_every == 0:
            wr, rank = evaluate(net, seeds=6, silent=True)
            if wr > best_wr:
                best_wr = wr
                save_model(net)
            print(f"  ppo round {rnd}: loss={loss:.3f} alive={alive_rate:.2f} "
                  f"eval_wr={wr:.2f} rank={rank:.2f} (collect {collect_t:.0f}s, train {train_t:.0f}s, "
                  f"pool={len(pool_eps)})", flush=True)
            # curriculum: medium -> hard
            if wr > 0.65 and skill == "medium":
                skill = "hard"
                print("  >>> curriculum: upgraded to HARD bots", flush=True)
            elif wr < 0.3 and skill == "hard":
                skill = "medium"
                print("  >>> curriculum: dropped back to MEDIUM bots", flush=True)
        else:
            print(f"  ppo round {rnd}: loss={loss:.3f} alive={alive_rate:.2f} "
                  f"(collect {collect_t:.0f}s, train {train_t:.0f}s)", flush=True)
    save_model(net)
    return net


# ================================================================ eval =======
def evaluate(net, seeds=6, silent=False):
    from sim.game5 import ClickSim5
    wins = 0
    ranks = []
    for seed in range(1, seeds + 1):
        game = ClickSim5(h=SIM["h"], w=SIM["w"], n_bots=SIM["n_bots"], seed=seed,
                         max_ticks=SIM["max_ticks"], clicks_per_tick=SIM["clicks_per_tick"],
                         bot_skill="hard")
        while game.tick < game.max_ticks:
            if not game.players[1].alive:
                break
            st = game.state_for(1)
            if not st.self_blob:
                break
            act, _ = _policy_action(net, st, game)
            actions = {1: game._clicks_for(act, SIM["clicks_per_tick"])}
            for pid in game._pids:
                if pid == 1 or not game.players[pid].alive:
                    continue
                actions[pid] = game._bot_clicks(pid)
            game.step(actions)
        alive = game.players[1].alive and (game.world == 1).sum() > 0
        wins += 1 if alive else 0
        areas = {pid: int((game.world == pid).sum()) for pid in game._pids}
        rank = 1 + sum(1 for pid, a in areas.items() if pid != 1 and a > areas[1])
        ranks.append(rank)
    wr = wins / seeds
    avg_rank = sum(ranks) / len(ranks)
    if not silent:
        print(f"EVAL (hard bots): alive-rate={wr:.2f} avg_rank={avg_rank:.2f}", flush=True)
    return wr, avg_rank


# ================================================================ main =======
def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    net = make_net().to(DEVICE)
    if MODEL_PT.exists():
        load_model(net)
    if stage in ("collect", "all"):
        stage_collect()
    if stage in ("vision", "all"):
        net = stage_vision(net)
        save_model(net)
    if stage in ("clone", "all"):
        net = stage_clone(net)
        save_model(net)
    if stage in ("ppo", "all"):
        net = stage_ppo(net, rounds=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
        save_model(net)
    evaluate(net, seeds=6)
    save_model(net)
    print(f"DONE. params={count_params(net)} model={MODEL_PT}", flush=True)


if __name__ == "__main__":
    main()
