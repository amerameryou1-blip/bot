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
try:
    from sim.game6 import ClickSim6 as ClickSim, map_slugs
    import sim.game6 as _g6
    _HAS_SIM6 = True
except Exception:
    from sim.game5 import ClickSim5 as ClickSim
    _HAS_SIM6 = False
from bot.planner import ClickAction

WEIGHTS = REPO / "weights" / "nn"
WEIGHTS.mkdir(parents=True, exist_ok=True)
DATASET = WEIGHTS / "dataset.npz"
MODEL_PT = WEIGHTS / "model.pt"
CFG_JSON = WEIGHTS / "config.json"

GRID = 16
CTX_DIM = 3


def _safe_device():
    """Pick cuda ONLY if it actually executes ops (P100 sm_60 is not supported
    by modern torch builds — torch.cuda.is_available() lies about that)."""
    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        a = torch.randn(16, 16, device="cuda")
        (a @ a).sum().item()
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as e:
        print(f"CUDA unusable ({str(e)[:90]}) — using CPU", flush=True)
        return torch.device("cpu")


DEVICE = _safe_device()
print(f"device: {DEVICE}", flush=True)

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

# sim params for training (v8: real maps + big lobbies by default)
SIM = dict(h=200, w=200, n_bots=int(os.environ.get("SIM_BOTS", "10")),
           max_ticks=int(os.environ.get("SIM_TICKS", "2400")),
           clicks_per_tick=14,
           map_slug=os.environ.get("SIM_MAP", "random"))


def _pick_map_slug():
    if not _HAS_SIM6:
        return None
    slugs = map_slugs()
    if not slugs:
        return None
    if SIM["map_slug"] and SIM["map_slug"] != "random":
        return SIM["map_slug"] if SIM["map_slug"] in slugs else slugs[0]
    return str(np.random.choice(slugs))


def _make_game(skill: str, seed: int, n_bots: int | None = None):
    nb = n_bots if n_bots is not None else getattr(_g6, "_CURR_BOTS", SIM["n_bots"]) if _HAS_SIM6 else None
    if nb is None:
        nb = SIM["n_bots"]
    if _HAS_SIM6:
        return ClickSim(n_bots=int(nb), seed=seed, max_ticks=SIM["max_ticks"],
                        clicks_per_tick=SIM["clicks_per_tick"], bot_skill=skill,
                        map_slug=_pick_map_slug())
    return ClickSim(h=SIM["h"], w=SIM["w"], n_bots=max(3, int(nb)), seed=seed,
                    max_ticks=SIM["max_ticks"], clicks_per_tick=SIM["clicks_per_tick"],
                    bot_skill=skill)


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
    json.dump({"grid": GRID, "context_dim": CTX_DIM, "classes": NUM_CLASSES,
               "params": count_params(net)},
              open(CFG_JSON, "w"), indent=2)


def load_model(net):
    """Shape-safe: an old 85k checkpoint can never crash a new 2M net."""
    if MODEL_PT.exists():
        try:
            net.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
            print("loaded checkpoint", flush=True)
        except Exception as e:
            print(f"checkpoint shape mismatch ({str(e)[:60]}) — fresh net",
                  flush=True)


# ================================================== 2026 optimizer recipe ===
class Muon(torch.optim.Optimizer):
    """Orthogonalized-momentum optimizer for 2D params (KellerJordan style).
    2026 evidence: beats AdamW on vision (arXiv 2605.24770), ~35% faster
    (PyTorch blog); the hybrid Muon(2D)+AdamW(1D) won small-scale practice
    (r/LocalLLaMA)."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay))

    @staticmethod
    def _ns(g, steps=5, eps=1e-7):
        orig = g.shape
        if g.dim() > 2:
            g = g.reshape(orig[0], -1)
        g = g / (g.norm() + eps)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            gram = g @ g.T
            g = a * g + (b * gram + c * gram @ gram) @ g
        return g.reshape(orig)

    @torch.no_grad()
    def step(self, closure=None):
        for grp in self.param_groups:
            lr = grp["lr"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                st["buf"].mul_(grp["momentum"]).add_(p.grad)
                if grp["weight_decay"]:
                    p.mul_(1 - lr * grp["weight_decay"])
                p.add_(self._ns(st["buf"]), alpha=-lr)


class HybridOpt:
    """Muon on matrices + AdamW on scalars; one .step()/.zero_grad() API and a
    loud NaN watchdog (skip the step, and if it keeps happening, fall back to
    pure AdamW — never ship NaN weights)."""

    def __init__(self, net, lr_muon=0.02, lr_adam=3e-4):
        mats = [p for p in net.parameters() if p.dim() >= 2 and p.requires_grad]
        rest = [p for p in net.parameters() if p.dim() < 2 and p.requires_grad]
        self.muon = Muon(mats, lr=lr_muon) if mats else None
        self.adam = torch.optim.AdamW(rest, lr=lr_adam) if rest else None
        self._nan_streak = 0
        self._degraded = False

    def zero_grad(self):
        if self.muon:
            self.muon.zero_grad()
        if self.adam:
            self.adam.zero_grad()

    def set_lr(self, lr_adam, lr_muon=None):
        if self.adam:
            for g in self.adam.param_groups:
                g["lr"] = lr_adam
        if self.muon and lr_muon:
            for g in self.muon.param_groups:
                g["lr"] = lr_muon

    def finite_ok(self, loss) -> bool:
        """Call BEFORE backward; loud watchdog with graceful degradation."""
        ok = bool(torch.isfinite(torch.as_tensor(loss)))
        if ok:
            self._nan_streak = 0
            return True
        self._nan_streak += 1
        print(f"[opt] NON-FINITE loss ({self._nan_streak} in a row) — skipping step",
              flush=True)
        if self._nan_streak >= 5 and not self._degraded and self.muon:
            print("[opt] LOUD: Muon unstable — degrading to AdamW-only", flush=True)
            self._degraded = True
            self.muon = None
        return False

    def step(self):
        if self.muon:
            self.muon.step()
        if self.adam:
            self.adam.step()


def make_optimizer(net):
    return HybridOpt(net)


def _augment(xb, yb=None):
    """Cheap 2026 vision recipe: random hflip + brightness jitter.
    (Muon gains scale with augmentation — arXiv 2605.24770.)"""
    if np.random.rand() < 0.5:
        xb = torch.flip(xb, dims=[-1])
        if yb is not None:
            yb = torch.flip(yb, dims=[-1])
    xb = xb * (0.88 + 0.24 * np.random.rand())
    return xb.clamp(0, 1), yb


# ========================================================== seed (distill) ==
def stage_seed(net, epochs=2, bs=256):
    """Distill the old 85k teacher into the new 2M net so it starts smart
    instead of random (teacher-student is the mature 2026 recipe)."""
    tp = WEIGHTS / "teacher.pt"
    if not tp.exists() or not DATASET.exists():
        print("[seed] no teacher.pt or dataset — skipping", flush=True)
        return net
    from nn.model import TerritoryNetSmall
    teacher = TerritoryNetSmall(grid=GRID, context_dim=CTX_DIM).to(DEVICE)
    try:
        teacher.load_state_dict(torch.load(tp, map_location=DEVICE))
    except Exception as e:
        print(f"[seed] teacher load failed ({str(e)[:60]}) — skipping", flush=True)
        return net
    teacher.eval()
    opt = make_optimizer(net)
    d = np.load(DATASET, mmap_mode="r")
    n = len(d["rgb"])
    for ep in range(epochs):
        net.train()
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            xb = torch.tensor(d["rgb"][i:i + bs], dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(d["seg"][i:i + bs], dtype=torch.int64, device=DEVICE)
            with torch.no_grad():
                t_seg, _, t_click, t_kind, _, _ = teacher.forward(xb, None, return_all=True)
            seg, local, click, kind, pct, _ = net.forward(xb, None, return_all=True)
            yb16 = F.interpolate(yb.float().unsqueeze(1), size=(GRID, GRID),
                                 mode="nearest").long().squeeze(1)
            loss = F.cross_entropy(seg, yb16, label_smoothing=0.05)
            loss = loss + F.kl_div(F.log_softmax(seg, 1), F.softmax(t_seg, 1),
                                   reduction="batchmean")
            loss = loss + F.kl_div(F.log_softmax(click, 1), F.softmax(t_click, 1),
                                   reduction="batchmean")
            loss = loss + F.kl_div(F.log_softmax(kind, 1), F.softmax(t_kind, 1),
                                   reduction="batchmean")
            if not opt.finite_ok(loss):
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  seed epoch {ep + 1}: loss={tot / max(nb, 1):.4f}", flush=True)
    # publish the distilled 2M brain where the RL loop picks it up, so the
    # loop starts from "teacher knowledge", not random weights
    try:
        rl_dir = WEIGHTS / "rl"
        rl_dir.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), rl_dir / "best.pt")
        json.dump({"ts": int(time.time()), "wr": 0.0, "rank": 99.0,
                   "source": "seed-distill"}, open(rl_dir / "best.json", "w"))
        print("[seed] published distilled brain to rl/best.pt", flush=True)
    except Exception as e:
        print(f"[seed] publish failed: {str(e)[:60]}", flush=True)
    return net


# ================================================================ collect ====
def stage_collect(seeds=None, bot_skill="medium", workers=None):
    seeds = seeds or int(os.environ.get("COLLECT_SEEDS", "28"))
    workers = workers or int(os.environ.get("WORKERS", "2"))
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
    opt = make_optimizer(net)
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
            xb, yb = _augment(xb, yb)
            seg, local, *_ = net.forward(xb, return_all=True)
            yb16 = F.interpolate(yb.float().unsqueeze(1), size=(GRID, GRID), mode="nearest").long().squeeze(1)
            loss = F.cross_entropy(seg, yb16, weight=seg_w, label_smoothing=0.05) \
                + F.mse_loss(local, cb) * 5.0
            if not opt.finite_ok(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  vision epoch {ep+1}: loss={tot/max(nb,1):.4f}", flush=True)
    return net


# ================================================================ clone ======
def stage_clone(net, epochs=8, bs=256, lr=1e-3):
    opt = make_optimizer(net)
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
            xb, _ = _augment(xb)
            click, kind, pct, _ = net.forward(xb, ctx)
            loss = F.cross_entropy(kind, kb, weight=kind_w, label_smoothing=0.05)
            mask = kb != 2
            if mask.any():
                loss = loss + F.cross_entropy(click[mask], cb[mask]) * 1.5
            am = kb == 1
            if am.any():
                loss = loss + F.mse_loss(pct[am], pb[am]) * 2.0
            if not opt.finite_ok(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  clone epoch {ep+1}: loss={tot/max(nb,1):.4f}", flush=True)
    return net


# ================================================================ PPO ========
NET_G = None  # worker-global net


def _init_worker(state_dict):
    global NET_G
    NET_G = make_net()
    NET_G.load_state_dict(state_dict)
    NET_G.eval()


def _policy_action(net, st, game):
    dev = next(net.parameters()).device
    rgb, _ = game.frame_tensor(1, size=64)
    # BUGFIX 2026-08-09: frame_tensor returns uint8 0-255; the net expects
    # 0..1 — every rollout/eval before this ran on 255-scale garbage input.
    x = torch.tensor(rgb.transpose(2, 0, 1)[None], dtype=torch.float32, device=dev) / 255.0
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


def _rollout_one(net, seed, skill, episodes=1, n_bots=None):
    """Play `episodes` matches; return a list of episode dicts."""
    eps = []
    for e in range(episodes):
        game = _make_game(skill, seed * 100 + e, n_bots=n_bots)
        ob_rgb, ob_ctx, k, c, p, lp, rw = [], [], [], [], [], [], []
        done = False
        while game.tick < game.max_ticks:
            if not game.players[1].alive:
                break
            st = game.state_for(1)
            if not st.self_blob:
                break
            act, pctv = _policy_action(net, st, game)
            dev = next(net.parameters()).device
            rgb, _ = game.frame_tensor(1, size=64)
            x = torch.tensor(rgb.transpose(2, 0, 1)[None], dtype=torch.float32, device=dev) / 255.0
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
            kills_before = game.players[1].kills
            actions = {1: game._clicks_for(act, SIM["clicks_per_tick"])}
            for pid in game._pids:
                if pid == 1 or not game.players[pid].alive:
                    continue
                actions[pid] = game._bot_clicks(pid)
            game.step(actions)
            area_after = int((game.world == 1).sum())
            reward = (area_after - area_before) / 2000.0  # territory growth
            kills_now = game.players[1].kills
            if kills_now > kills_before:
                reward += 2.0 * (kills_now - kills_before)  # KILL BONUS (v8)
            if game.players[1].alive:
                reward += 0.005  # small survival tick (v8: much smaller than v7)
                if area_after <= area_before:
                    reward -= 0.002  # idle penalty: standing still costs you
            else:
                reward -= 1.0
                done = True
            rw.append(reward)
            if done or len(rw) >= 250:  # bound episode length (memory)
                break
        if len(rw) < 3:
            import gc; del ob_rgb, ob_ctx, k, c, p, lp, rw; gc.collect()
            continue
        alive_final = game.players[1].alive
        final_area = int((game.world == 1).sum())
        # LAST SURVIVOR = the goal: big win bonus, tiny area bonus
        alive_list = [pid for pid in game._pids if game.players[pid].alive]
        is_winner = alive_final and len(alive_list) == 1
        rw[-1] += (5.0 if is_winner else 0.0) + min(final_area / 20000.0, 1.0) * 0.3
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
    """PPO update. FIX (v7): advantages are normalized ACROSS the whole batch,
    not per-episode — per-episode normalization zeros the gradient whenever an
    episode's rewards are near-constant (tiny growth ticks), which is exactly
    the v5 'loss=0.000, nothing learned' failure. Rewards are scaled x10 so
    per-tick growth is visible vs the survival tick."""
    net.train()
    tot_loss, nb = 0.0, 0
    ep_adv, ep_ret = [], []
    for ep in episodes:
        T = len(ep["reward"])
        rw = np.array(ep["reward"], dtype=np.float32) * 10.0
        with torch.no_grad():
            # ep["rgb"] is already 0..1 (rollout/unpack) — do NOT /255 again
            xb = torch.tensor(ep["rgb"], dtype=torch.float32, device=DEVICE)
            cxb = torch.tensor(ep["ctx"], device=DEVICE)
            _, _, _, vals = net.forward(xb, cxb)
            vals = vals.cpu().numpy()
        adv = np.zeros(T); gae = 0.0
        for t in reversed(range(T)):
            nxt = vals[t + 1] if t + 1 < T else (1.0 if ep["alive"] else 0.0)
            delta = rw[t] + gamma * nxt - vals[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        ret = np.array([rw[t] + (vals[t + 1] if t + 1 < T else (1.0 if ep["alive"] else 0.0)) * gamma
                        for t in range(T)], dtype=np.float32)
        ep_adv.append(adv); ep_ret.append(ret)
    # GLOBAL advantage normalization across the whole batch
    flat = np.concatenate(ep_adv) if ep_adv else np.zeros(1)
    if flat.std() > 1e-6:
        flat = (flat - flat.mean()) / (flat.std() + 1e-8)
    off = 0
    for adv in ep_adv:
        adv[:] = flat[off:off + len(adv)]
        off += len(adv)
    # 2026 PPO hygiene: shuffle episode order every epoch + entropy anneal
    # (explore early, commit late) + non-finite loss watchdog.
    items = list(zip(episodes, ep_adv, ep_ret))
    for e_i in range(epochs):
        np.random.shuffle(items)
        ent_coef = 0.05 - 0.04 * (e_i / max(epochs - 1, 1))
        for ep, adv, ret in items:
            xb = torch.tensor(ep["rgb"], dtype=torch.float32, device=DEVICE)
            cxb = torch.tensor(ep["ctx"], device=DEVICE)
            kb = torch.tensor(ep["kind"], device=DEVICE)
            cb = torch.tensor(ep["cell"], device=DEVICE)
            pb = torch.tensor(ep["pct"], device=DEVICE)
            old_lp = torch.tensor(ep["logp"], device=DEVICE)
            adv_t = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
            ret_t = torch.tensor(ret, device=DEVICE)
            click, kind, pct, value = net.forward(xb, cxb)
            logpk = F.log_softmax(kind, dim=1)
            logpc = F.log_softmax(click, dim=1)
            lp = logpk.gather(1, kb.unsqueeze(1)).squeeze(1)
            nb_mask = (kb != 2).float()
            cell_lp = logpc.gather(1, cb.unsqueeze(1)).squeeze(1)
            lp = lp + cell_lp * nb_mask
            ratio = torch.exp(lp - old_lp)
            pg = -torch.min(ratio * adv_t, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t).mean()
            vf = F.mse_loss(value, ret_t)
            ent = -torch.mean(logpk * logpk.exp())
            loss = pg + 0.5 * vf - ent_coef * ent
            if not opt.finite_ok(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item(); nb += 1
    return tot_loss / max(nb, 1)


# ============================================================ real data ====
REAL_NPZ = WEIGHTS / "real_vision.npz"


def stage_real(net, epochs=12, bs=128, lr=3e-5):
    """Fine-tune vision (segmentation) + click head on REAL frames.

    Real data comes from scripts/label_real.py (recordings + screenshots):
      rgb (N,64,64,3), labels (N,64,64), kind/cell/pct (click labels).
    Loud gates: report per-class pixel accuracy on a held-out 10% split.
    """
    if not REAL_NPZ.exists():
        print(f"[real] no {REAL_NPZ} — skipping (run scripts/label_real.py first)", flush=True)
        return net
    d = np.load(REAL_NPZ)
    rgb_all = d["rgb"]
    if rgb_all.dtype != np.float32:
        rgb_all = rgb_all.astype(np.float32) / 255.0  # uint8 -> 0..1
    n = len(rgb_all)
    idx = np.random.RandomState(0).permutation(n)
    n_val = max(1, n // 10)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    has_clicks = len(d["kind"]) > 0

    print(f"[real] {n} real frames ({n_val} val), clicks={has_clicks}", flush=True)

    def classify_acc(net, subset, n_batches=8):
        net.eval()
        correct = np.zeros(5); total = np.zeros(5)
        with torch.no_grad():
            for i in range(0, len(subset), bs):
                ii = subset[i:i+bs]
                xb = torch.tensor(rgb_all[ii].transpose(0, 3, 1, 2), dtype=torch.float32, device=DEVICE)
                lb = torch.tensor(d["labels"][ii], dtype=torch.int64, device=DEVICE)
                seg, *_ = net.forward(xb, None, return_all=True)
                # seg is at GRID res; downsample labels to match
                lb_g = F.interpolate(lb.float().unsqueeze(1), size=(GRID, GRID), mode="nearest").long().squeeze(1)
                pred = seg.argmax(1)
                for c in range(5):
                    m = lb_g == c
                    total[c] += int(m.sum())
                    correct[c] += int((pred[m] == c).sum())
        return {c: (correct[c] / total[c] if total[c] else float("nan")) for c in range(5)}

    opt = make_optimizer(net)
    # v14 class weights: sqrt-inverse-frequency, CAPPED [0.2, 0.8].
    # v13 ran the old 1/freq weights [0.2,0.2,1.0,0.2,0.2] — me=1.0 starved
    # water/enemy/ui at 0.2 -> me-overfit, water/enemy never learned.
    dist = np.bincount(d["labels"].ravel(), minlength=5).astype(np.float64)
    dist = np.clip(dist, 1, None)
    freq = dist / dist.sum()
    inv = 1.0 / np.sqrt(freq)
    inv = inv / inv.max()
    inv = np.clip(inv, 0.2, 0.8)
    class_w = torch.tensor(inv, dtype=torch.float32, device=DEVICE)
    print(f"[real] sqrt-inverse-freq class weights (capped 0.2-0.8): {[round(float(w), 3) for w in inv]}", flush=True)

    # rare-class frame oversampling: frames with me/enemy pixels sampled more
    me_cnt = (d["labels"] == 2).sum(axis=(1, 2)).astype(np.float64)
    en_cnt = (d["labels"] == 3).sum(axis=(1, 2)).astype(np.float64)
    frame_w = 1.0 + 4.0 * me_cnt / 128.0 + 1.5 * en_cnt / 128.0
    frame_w = frame_w / frame_w.mean()

    # sim mixing: every 2nd batch comes from the sim dataset (mmap) so the
    # backbone doesn't forget the sim domain it was pretrained on
    sim = None
    if DATASET.exists():
        try:
            sim = np.load(DATASET, mmap_mode="r")
            print(f"[real] sim mixing on: {len(sim['rgb'])} sim frames", flush=True)
        except Exception as e:
            print(f"[real] sim mixing unavailable: {e}", flush=True)
    n_sim = len(sim["rgb"]) if sim is not None else 0

    import math
    lr_min = lr * 0.1
    for ep in range(epochs):
        lr_t = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * ep / max(epochs, 1)))
        opt.set_lr(lr_t)
        net.train()
        tot, nb = 0.0, 0
        order = np.random.choice(len(tr_i), size=len(tr_i),
                                 p=frame_w[tr_i] / frame_w[tr_i].sum())
        for i in range(0, len(order), bs):
            ii = order[i:i + bs]
            if sim is not None and n_sim and (nb % 2) == 1:
                # sim batch (already 0..1, NCHW)
                si = np.random.randint(0, n_sim, size=len(ii))
                xb = torch.tensor(sim["rgb"][si], dtype=torch.float32, device=DEVICE)
                lb = torch.tensor(sim["seg"][si], dtype=torch.int64, device=DEVICE)
            else:
                # real batch — MUST be the /255 normalized rgb_all (the old
                # d["rgb"] uint8 0..255 wrecked transfer from the sim stage)
                xb = torch.tensor(rgb_all[ii].transpose(0, 3, 1, 2), dtype=torch.float32, device=DEVICE)
                lb = torch.tensor(d["labels"][ii], dtype=torch.int64, device=DEVICE)
            xb, lb = _augment(xb, lb)
            seg, _, click, kind, pct, _ = net.forward(xb, None, return_all=True)
            lb_g = F.interpolate(lb.float().unsqueeze(1), size=(GRID, GRID), mode="nearest").long().squeeze(1)
            loss = F.cross_entropy(seg, lb_g, weight=class_w, label_smoothing=0.05)
            if not opt.finite_ok(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        acc = classify_acc(net, val_i)
        print(f"[real] epoch {ep + 1} (lr={lr_t:.2e}): seg_loss={tot / max(nb, 1):.4f} | "
              f"water={acc[0]:.2f} neutral={acc[1]:.2f} me={acc[2]:.2f} enemy={acc[3]:.2f} ui={acc[4]:.2f}",
              flush=True)

    # loud gates (vision quality on REAL frames) — the user's ship gates:
    # water>=97%, me>=90%, enemy>=85%, ui>=98%. Integer keys match
    # classify_acc() — the old string keys ALWAYS passed (silent-failure bug).
    acc = classify_acc(net, val_i)
    gates = {0: 0.97, 2: 0.90, 3: 0.85, 4: 0.98}
    fails = [k for k, v in gates.items() if acc.get(k) is not None and acc[k] < v and not np.isnan(acc[k])]
    if fails:
        names = {0: "water", 2: "me", 3: "enemy", 4: "ui"}
        print(f"[real] VISION GATE FAILED for {[names.get(k, k) for k in fails]} "
              f"(acc={ {names.get(k, k): round(acc[k], 3) for k in fails} }) — "
              f"REAL VISION NOT TRUSTWORTHY", flush=True)
        raise RuntimeError("real vision gate failed")
    else:
        print(f"[real] VISION GATE PASSED on real frames "
              f"(water={acc[0]:.3f} me={acc[2]:.3f} enemy={acc[3]:.3f} ui={acc[4]:.3f})", flush=True)

    # click clone on real clicks (if present)
    if has_clicks and len(d["kind"]) > 0:
        print(f"[real] cloning click head on {len(d['kind'])} real clicks", flush=True)
        kind_w = torch.tensor([1.0, 4.0, 1.5], device=DEVICE)
        for ep in range(6):
            net.train()
            order = np.random.permutation(len(d["kind"]))
            tot, nb = 0.0, 0
            for i in range(0, len(order), bs):
                ii = order[i:i+bs]
                xb = torch.tensor(rgb_all[ii].transpose(0, 3, 1, 2), dtype=torch.float32, device=DEVICE)
                kb = torch.tensor(d["kind"][ii], dtype=torch.int64, device=DEVICE)
                cb = torch.tensor(d["cell"][ii], dtype=torch.int64, device=DEVICE)
                pb = torch.tensor(d["pct"][ii], dtype=torch.float32, device=DEVICE)
                ctx = torch.zeros(xb.shape[0], CTX_DIM, device=DEVICE)
                click, kind, pct, _ = net.forward(xb, ctx)
                loss = F.cross_entropy(kind, kb, weight=kind_w)
                mask = kb != 2
                if mask.any():
                    loss = loss + F.cross_entropy(click[mask], cb[mask]) * 1.5
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            print(f"[real] click clone epoch {ep + 1}: loss={tot / max(nb, 1):.4f}", flush=True)
    return net


def stage_ppo(net, rounds=None, episodes_per_worker=1, workers=None, lr=1e-4, eval_every=5,
              pool_cap=80):
    rounds = rounds or int(os.environ.get("PPO_ROUNDS", "40"))
    ncores = os.cpu_count() or 2
    workers = workers or int(os.environ.get("WORKERS", str(ncores - 1 if ncores > 1 else 1)))
    opt = make_optimizer(net)
    import multiprocessing as mp
    pool_eps = []
    skill = "easy"  # curriculum: learn to SURVIVE first (v8: + enemy count)
    n_bots = 2
    best_wr = 0.0
    for rnd in range(1, rounds + 1):
        net.eval()
        state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
        t0 = time.time()
        # enemy count for this round (global knob for the pool workers)
        _g6._CURR_BOTS = n_bots
        if workers <= 1:
            # inline (no pool) — used on low-RAM boxes and as a fallback
            for w in range(episodes_per_worker):
                pool_eps.extend(_rollout_one(net, 1000 + rnd * 10 + w, skill, 1))
        else:
            with mp.Pool(workers, initializer=_init_worker, initargs=(state_dict,)) as pool:
                tasks = [(1000 + rnd * 10 + w, skill, episodes_per_worker) for w in range(workers)]
                for res in pool.imap_unordered(_rollout_worker, tasks):
                    pool_eps.extend(res)
        collect_t = time.time() - t0
        # cap pool (drop oldest)
        if len(pool_eps) > pool_cap:
            pool_eps = pool_eps[-pool_cap:]
        import gc
        gc.collect()
        t0 = time.time()
        loss = _train_ppo_batch(net, opt, pool_eps)
        train_t = time.time() - t0
        alive_rate = sum(e["alive"] for e in pool_eps[-workers * episodes_per_worker:]) / max(workers * episodes_per_worker, 1)
        if rnd % eval_every == 0:
            wr, rank = evaluate(net, seeds=6, silent=True)
            if wr > best_wr:
                best_wr = wr
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"  ppo round {rnd}: loss={loss:.3f} alive={alive_rate:.2f} "
                  f"eval_wr={wr:.2f} rank={rank:.2f} (collect {collect_t:.0f}s, train {train_t:.0f}s, "
                  f"pool={len(pool_eps)})", flush=True)
            # v8 curriculum: skill easy->medium->hard AND enemy count
            # 2 -> 4 -> 8 -> SIM["n_bots"], driven by alive rate
            if alive_rate > 0.35 and (skill == "easy" or n_bots < SIM["n_bots"]):
                if skill == "easy":
                    skill = "medium"
                    print("  >>> curriculum: EASY -> MEDIUM", flush=True)
                if n_bots < SIM["n_bots"]:
                    n_bots = min(SIM["n_bots"], max(n_bots + 1, int(n_bots * 1.6)))
                    print(f"  >>> curriculum: enemies {max(0, n_bots - 1)} -> {n_bots}", flush=True)
            if alive_rate > 0.6 and skill == "medium":
                skill = "hard"
                print("  >>> curriculum: MEDIUM -> HARD", flush=True)
            if alive_rate < 0.3 and skill == "hard":
                skill = "medium"
                print("  >>> curriculum: HARD -> MEDIUM", flush=True)
        else:
            print(f"  ppo round {rnd}: loss={loss:.3f} alive={alive_rate:.2f} "
                  f"(collect {collect_t:.0f}s, train {train_t:.0f}s)", flush=True)
    # restore + save the BEST eval model (the unconditional save_model at the
    # end used to overwrite it with the LAST round's net)
    if best_wr > 0:
        net.load_state_dict(best_state)
        print(f"[ppo] restoring best eval model (wr={best_wr}) before save", flush=True)
    save_model(net)
    return net


# ================================================================ eval =======
def evaluate(net, seeds=6, silent=False):
    net.eval()
    wins = 0
    survived = 0
    ranks = []
    for seed in range(1, seeds + 1):
        game = _make_game("mixed", seed, n_bots=SIM["n_bots"])
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
        alive_list = [pid for pid in game._pids if game.players[pid].alive]
        # HONEST last-survivor gate (2026-08-08 fix, found by vision agent):
        # "win" = we are the ONLY player alive. The old code counted "still
        # alive at timeout" as a win, inflating win-rate in unfinished games.
        is_last = bool(alive) and len(alive_list) == 1
        wins += 1 if is_last else 0
        survived += 1 if alive else 0
        areas = {pid: int((game.world == pid).sum()) for pid in game._pids}
        rank = 1 + sum(1 for pid, a in areas.items() if pid != 1 and a > areas[1])
        ranks.append(rank)
    wr = wins / seeds
    alive_rate = survived / seeds
    avg_rank = sum(ranks) / len(ranks)
    if not silent:
        print(f"EVAL ({SIM['n_bots']}-player mixed lobby): "
              f"LAST-SURVIVOR win-rate={wr:.2f} alive-rate={alive_rate:.2f} "
              f"avg_rank={avg_rank:.2f}", flush=True)
    return wr, avg_rank


# ================================================================ main =======
def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    net = make_net().to(DEVICE)
    if MODEL_PT.exists():
        load_model(net)
    if stage == "eval":
        wr, rank = evaluate(net, seeds=6)
        print(f"EVAL(wr={wr:.2f}, rank={rank:.2f}) model={MODEL_PT}", flush=True)
        return
    if stage in ("collect", "all"):
        stage_collect()
    if stage in ("seed", "all"):
        net = stage_seed(net)
        save_model(net)
    if stage in ("vision", "all"):
        net = stage_vision(net)
        save_model(net)
    if stage in ("clone", "all"):
        net = stage_clone(net)
        save_model(net)
    if stage in ("real", "all"):
        net = stage_real(net)
        save_model(net)
    if stage in ("ppo", "all"):
        net = stage_ppo(net, rounds=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
        save_model(net)
    evaluate(net, seeds=6)
    save_model(net)
    print(f"DONE. params={count_params(net)} model={MODEL_PT}", flush=True)


if __name__ == "__main__":
    main()
