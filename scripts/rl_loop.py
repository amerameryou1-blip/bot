#!/usr/bin/env python3
"""Continuous sim-RL loop — HANDOFF §14 v1 (worker + trainer modes).

The loop never stops between Kaggle session caps:
  * WORKERS  play sim matches with the latest checkpoint (the stochastic
    policy IS the exploration), push trajectory shards, pick up newer
    checkpoints when they appear.
  * TRAINER  pulls new shards -> PPO -> loud last-survivor eval -> pushes a
    new checkpoint when it improves.

Credential-optional: with HF_TOKEN everything syncs through
amer224/territorial-bot-data (rl/shards/, rl/ckpt_*.pt, rl/best.json);
without it the loop runs fully LOCAL (weights/nn/rl/) so the sandbox can
develop and smoke-test it.

Usage:
  python3 scripts/rl_loop.py worker  [--hours 8] [--skill medium] [--n-bots 8]
  python3 scripts/rl_loop.py trainer [--hours 8]
Env: HF_TOKEN, EP_PER_SHARD (8), SHARD_SLEEP_S (20), TRAIN_EPOCHS (4),
     SHIP_WR (0.30)
"""
import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch

import train_nn as T  # reuses _rollout_one / _train_ppo_batch / evaluate / net

HF_DATASET = "amer224/territorial-bot-data"
RL = T.WEIGHTS / "rl"
SHARDS = RL / "shards"
DONE = RL / "done"
for d in (RL, SHARDS, DONE):
    d.mkdir(parents=True, exist_ok=True)
BEST_PT = RL / "best.pt"
BEST_JSON = RL / "best.json"


def log(msg):
    print(f"[rl {time.strftime('%H:%M:%S')}] {msg}", flush=True)


_LAST_HB_UP = 0.0


def _hb(phase, t_it=None, **extra):
    """v6 telemetry: heartbeat with PHASE per sub-step. Upload throttled to
    once per 2 min (was every call -> ~22 commits/h eating the HF budget)."""
    global _LAST_HB_UP
    payload = {"ts": int(time.time()), "phase": phase}
    if t_it is not None:
        payload["iter_s"] = int(time.time() - t_it)
    payload.update(extra)
    if time.time() - _LAST_HB_UP < 120:
        return
    try:
        api_h, tok_h = _hf_api()
        if not api_h:
            return
        hb = RL / "hb.json"
        json.dump(payload, open(hb, "w"))
        api_h.upload_file(path_or_fileobj=str(hb),
                          path_in_repo="rl/trainer_heartbeat.json",
                          repo_id=HF_DATASET, repo_type="dataset", token=tok_h)
        _LAST_HB_UP = time.time()
    except Exception:
        pass


def _hf_api():
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        return None, None
    try:
        from huggingface_hub import HfApi
        return HfApi(token=tok), tok
    except Exception as e:
        log(f"HF unavailable ({str(e)[:80]}) — LOCAL mode")
        return None, None


# ------------------------------------------------------------ checkpoints ---
def best_meta():
    if BEST_JSON.exists():
        try:
            return json.load(open(BEST_JSON))
        except Exception:
            pass
    return {"wr": -1.0, "rank": 99.0, "ts": 0, "source": "none"}


def _try_load(net, path):
    """Shape-safe: returns True only if the checkpoint fits this net."""
    try:
        net.load_state_dict(torch.load(path, map_location=T.DEVICE))
        return True
    except Exception as e:
        log(f"ckpt {Path(path).name} mismatch ({str(e)[:50]}) — skipped")
        return False


def _read_best_json(api, tok):
    """best.json with validation + forced retry (a cached half-written copy
    made a trainer restart from the wrong brain on 2026-08-14)."""
    for force in (False, True):
        try:
            p = api.hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                                    filename="rl/best.json", token=tok,
                                    force_download=force)
            return json.load(open(p))
        except Exception:
            continue
    return None


def load_latest_net():
    """Newest checkpoint: HF rl/best.json+ckpt -> ckpt-list fallback ->
    local best.pt -> model.pt. All loads shape-safe."""
    net = T.make_net().to(T.DEVICE)
    api, tok = _hf_api()
    if api:
        remote = _read_best_json(api, tok)
        if remote is not None:
            try:
                p = api.hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                                        filename=f"rl/ckpt_{remote['ts']}.pt",
                                        token=tok)
                if _try_load(net, p):
                    log(f"pulled newer checkpoint ts={remote['ts']} "
                        f"wr={remote.get('wr')}")
                    return net, remote
            except Exception as e:
                log(f"checkpoint pull skipped ({str(e)[:60]})")
        else:
            # best.json unreadable -> newest ckpt file on the hub by name
            try:
                cks = [f for f in api.list_repo_files(HF_DATASET,
                       repo_type="dataset", token=tok)
                       if f.startswith("rl/ckpt_")]
                if cks:
                    newest = max(cks, key=lambda f: int(f[:-3].split("_")[1]))
                    p = api.hf_hub_download(repo_id=HF_DATASET,
                                            repo_type="dataset",
                                            filename=newest, token=tok)
                    if _try_load(net, p):
                        log(f"best.json unreadable — fell back to {newest}")
                        return net, {"ts": int(newest[:-3].split("_")[1]),
                                     "wr": 0.0, "rank": 9.5,
                                     "source": "ckpt-fallback"}
            except Exception as e:
                log(f"ckpt fallback failed ({str(e)[:50]})")
    if BEST_PT.exists() and _try_load(net, BEST_PT):
        return net, best_meta()
    T.load_model(net)  # weights/nn/model.pt (shape-safe inside)
    return net, best_meta()


def save_checkpoint(net, wr, rank):
    ts = int(time.time())
    torch.save(net.state_dict(), BEST_PT)
    torch.save(net.state_dict(), RL / f"ckpt_{ts}.pt")
    meta = {"ts": ts, "wr": float(wr), "rank": float(rank), "source": "trainer"}
    json.dump(meta, open(BEST_JSON, "w"), indent=2)
    api, tok = _hf_api()
    if api:
        try:
            api.upload_file(path_or_fileobj=str(RL / f"ckpt_{ts}.pt"),
                            path_in_repo=f"rl/ckpt_{ts}.pt",
                            repo_id=HF_DATASET, repo_type="dataset", token=tok)
            api.upload_file(path_or_fileobj=str(BEST_JSON),
                            path_in_repo="rl/best.json",
                            repo_id=HF_DATASET, repo_type="dataset", token=tok)
            log(f"checkpoint ts={ts} wr={wr:.2f} -> HF")
        except Exception as e:
            log(f"checkpoint HF upload failed ({str(e)[:60]}) — local only")
    return ts


# ------------------------------------------------------------------ shards ---
def pack_episodes(episodes, seed0):
    """uint8 rgb (v14 lesson: store small, divide at train time)."""
    keys = ["rgb", "ctx", "kind", "cell", "pct", "logp", "reward", "alive"]
    out = {k: [] for k in keys}
    lens = []
    for ep in episodes:
        lens.append(len(ep["reward"]))
        out["rgb"].append((np.asarray(ep["rgb"]) * 255).astype(np.uint8))
        out["ctx"].append(np.asarray(ep["ctx"], dtype=np.float32))
        out["kind"].append(np.asarray(ep["kind"], dtype=np.int64))
        out["cell"].append(np.asarray(ep["cell"], dtype=np.int64))
        out["pct"].append(np.asarray(ep["pct"], dtype=np.float32))
        out["logp"].append(np.asarray(ep["logp"], dtype=np.float32))
        out["reward"].append(np.asarray(ep["reward"], dtype=np.float32))
        out["alive"].append(np.int64(1 if ep["alive"] else 0))
    return {
        "rgb": np.concatenate(out["rgb"]),
        "ctx": np.concatenate(out["ctx"]),
        "kind": np.concatenate(out["kind"]),
        "cell": np.concatenate(out["cell"]),
        "pct": np.concatenate(out["pct"]),
        "logp": np.concatenate(out["logp"]),
        "reward": np.concatenate(out["reward"]),
        "alive": np.array(out["alive"], dtype=np.int64),
        "lens": np.array(lens, dtype=np.int64),
        "seed0": np.int64(seed0),
    }


def unpack_episodes(shard):
    eps = []
    off = 0
    rgb = shard["rgb"]
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32) / 255.0
    for i, L in enumerate(shard["lens"]):
        eps.append({
            "rgb": rgb[off:off + L],
            "ctx": shard["ctx"][off:off + L],
            "kind": shard["kind"][off:off + L],
            "cell": shard["cell"][off:off + L],
            "pct": shard["pct"][off:off + L],
            "logp": shard["logp"][off:off + L],
            "reward": shard["reward"][off:off + L],
            "alive": bool(shard["alive"][i]),
        })
        off += L
    return eps


def push_shards_batch(folder: Path) -> bool:
    """ONE commit for all pending local shards (HF caps commits at
    128/h — per-shard commits were saturating the account budget)."""
    files = sorted(folder.glob("shard_*.npz"))
    if not files:
        return
    api, tok = _hf_api()
    if not api:
        return
    try:
        api.upload_folder(folder_path=str(folder),
                          path_in_repo="rl/" + folder.name,
                          repo_id=HF_DATASET, repo_type="dataset", token=tok,
                          ignore_patterns="*.partial")
        log(f"{len(files)} shards -> HF (1 commit)")
        for f in files:
            f.unlink()
        return True
    except Exception as e:
        log(f"batch upload failed ({str(e)[:60]}) — kept local, backoff")
        return False


# ------------------------------------------------------------------ worker ---
def mode_worker(hours, skill, n_bots):
    t_end = time.time() + hours * 3600
    ep_per_shard = int(os.environ.get("EP_PER_SHARD", "8"))
    sleep_s = float(os.environ.get("SHARD_SLEEP_S", "20"))
    flush_s = float(os.environ.get("FLUSH_S", "1200"))  # 1 HF commit / 20 min
    last_flush = time.time()
    shard_i = 0
    over_cap = False
    while time.time() < t_end:
        net, meta = load_latest_net()
        net.eval()
        # 2026 data-gen rule (arXiv 2603.24202): ENVIRONMENT DIVERSITY beats
        # scale; medium difficulty is the sweet spot (easy->overfit,
        # hard->sparse rewards). Randomize lobby per shard.
        sk = str(np.random.choice(["medium"] * 3 + ["easy", "hard"]))
        nb = int(np.random.choice([8, 12, 16, 24, 32]))  # chaotic but survivable
        episodes, seed0 = [], int(time.time()) % 100000
        for s in range(ep_per_shard):
            episodes += T._rollout_one(net, seed0 + s * 7 + 1, sk, 1, n_bots=nb)
        if not episodes:
            log("no episodes collected — sleeping")
            time.sleep(sleep_s)
            continue
        wins = 0
        for ep in episodes:
            if ep["alive"]:
                wins += 1  # note: alive, not last-survivor (workers are cheap)
        path = SHARDS / f"shard_rl_{int(time.time())}_{os.getpid()}_{shard_i}_{sk}_{nb}.npz"
        np.savez_compressed(path, **pack_episodes(episodes, seed0))
        shard_i += 1
        log(f"worker shard {path.name}: {len(episodes)} episodes "
            f"({sum(e['reward'].size for e in episodes)} steps) vs ckpt "
            f"wr={meta.get('wr', -1):.2f} alive={wins}/{len(episodes)}")
        # storage hygiene: stop uploading when the HF queue is too big
        # (trainer deletes consumed shards; cap bounds the backlog)
        if shard_i % 10 == 1:
            api0, tok0 = _hf_api()
            if api0:
                try:
                    n = sum(1 for _ in api0.list_repo_tree(
                        HF_DATASET, "rl/shards", repo_type="dataset", token=tok0))
                    over_cap = n > int(os.environ.get("HF_SHARD_CAP", "1200"))
                    if over_cap:
                        log(f"HF shard backlog {n} > cap — skipping uploads")
                except Exception:
                    pass
        if over_cap:
            loc = sorted(SHARDS.glob("shard_rl_*.npz"))
            for f in loc[:-30]:
                f.unlink()  # backlog capped: drop oldest local experience
        elif time.time() - last_flush > flush_s:
            if push_shards_batch(SHARDS):
                flush_s = 1200.0
            else:
                flush_s = min(flush_s * 2, 7200.0)   # quiet protocol
            last_flush = time.time()
        time.sleep(sleep_s)
    push_shards_batch(SHARDS)
    log("worker hours exhausted — exiting")


# ----------------------------------------------------------------- trainer ---
def mode_trainer(hours):
    t_end = time.time() + hours * 3600
    epochs = int(os.environ.get("TRAIN_EPOCHS", "4"))
    ship_wr = float(os.environ.get("SHIP_WR", "0.30"))
    api, tok = _hf_api()
    best = best_meta()
    # seed best: hub best.json first (lineage!), v14 eval only if hub empty
    if best["wr"] < 0:
        api0, tok0 = _hf_api()
        remote = _read_best_json(api0, tok0) if api0 else None
        if remote and remote.get("wr", -1) >= 0:
            best = remote
            log(f"seeded best from HUB: wr={best['wr']:.2f} "
                f"rank={best['rank']:.2f} ts={best['ts']}")
        elif T.MODEL_PT.exists():
            net0 = T.make_net().to(T.DEVICE)
            T.load_model(net0)
            wr, rank = T.evaluate(net0, seeds=6, silent=True)
            best = {"ts": int(time.time()), "wr": wr, "rank": rank,
                    "source": "v14-seed"}
            json.dump(best, open(BEST_JSON, "w"), indent=2)
            log(f"seeded best from v14 checkpoint: wr={wr:.2f} rank={rank:.2f}")
    net = None   # created on first training iter, then kept in-memory
    opt = None
    while time.time() < t_end:
        t_it = time.time()
        # heartbeat FIRST, before any heavy work (v6)
        _hb("iter_start", t_it)
        log(f"[hb] iter start (pending local={len(list(SHARDS.glob('shard_*.npz')))})")
        if api:
            try:
                files = api.list_repo_files(HF_DATASET, repo_type="dataset", token=tok)
                # v1 shard_rl_* pool is RETIRED — never look at it again
                # (ghost broken symlinks spun the trainer in a 2s loop,
                #  2026-08-12)
                new2 = [f for f in files
                        if f.startswith("rl/shards_v2/") and f.endswith(".npz")
                        and not (SHARDS_V2 / Path(f).name).exists()
                        and not (DONE_V2 / Path(f).name).exists()]
                for f in new2[:8]:
                    p = api.hf_hub_download(repo_id=HF_DATASET,
                                            repo_type="dataset",
                                            filename=f, token=tok)
                    # copyfile, not move: hf returns cache SYMLINKS; moving
                    # them left broken links that poisoned the glob
                    shutil.copyfile(p, str(SHARDS_V2 / Path(f).name))
                if new2:
                    log(f"pulled {len(new2)} HD shards from HF")
            except Exception as e:
                log(f"shard pull skipped ({str(e)[:60]})")
        for p in SHARDS_V2.glob("shard_v2_*.npz"):
            if p.is_symlink() or not p.exists():
                p.unlink(missing_ok=True)   # broken links from old versions
        pending_v2 = sorted(SHARDS_V2.glob("shard_v2_*.npz"))
        if not pending_v2:
            time.sleep(30)
            continue
        episodes = []
        for p in pending_v2:
            try:
                episodes += unpack_v2_episodes(np.load(p))
            except Exception as e:
                log(f"bad v2 shard {p.name}: {e}")
        if not episodes:
            for p in pending_v2:
                p.rename(DONE_V2 / p.name)
            time.sleep(30)
            continue
        # cap the PPO batch: 2M-param PPO is heavy on CPU; more iterations
        # with fresher data beats giant slow ones (measured 2026-08-09)
        cap = int(os.environ.get("EP_CAP", "24"))
        if len(episodes) > cap:
            idx = np.random.choice(len(episodes), cap, replace=False)
            episodes = [episodes[i] for i in sorted(idx)]
        if net is None:
            # load ONCE; then train continuously in-memory. Reloading the last
            # SAVED net every round (old code) discarded every round that
            # didn't beat the publish gate -> amnesia treadmill, no progress.
            net, _ = load_latest_net()
            opt = T.make_optimizer(net)   # Muon+AdamW hybrid, NaN-watchdogged
        _hb("training", t_it, eps=len(episodes))
        log(f"[hb] training on {len(episodes)} episodes "
            f"({time.time()-t_it:.0f}s since iter start)")
        loss = T._train_ppo_batch(net, opt, episodes, epochs=epochs)
        wr, rank = T.evaluate(net, seeds=int(os.environ.get("EVAL_SEEDS", "4")),
                              silent=True)
        log(f"[hb] trained loss={loss:.3f} wr={wr:.2f} rank={rank:.2f} "
            f"({time.time()-t_it:.0f}s)")
        msg = (f"trainer iter: shards={len(pending_v2)} episodes={len(episodes)} "
               f"loss={loss:.3f} | LAST-SURVIVOR wr={wr:.2f} rank={rank:.2f} "
               f"(best {best['wr']:.2f})")
        # 2026-08-14 fix: old gate `wr >= best.wr` saved on TIES -> with the
        # wr=0 plateau every round "won", flooding HF with a checkpoint every
        # ~5 min and burning the commit budget workers need. Now: save only on
        # a strict wr gain, or (at the wr=0 plateau) a rank jump >= 1.5, which
        # is beyond 4-seed eval noise.
        # 2026-08-14b: observed 8-seed rank swings of ±3.5 -> margin was BELOW
        # the noise (publishes could be lottery). Now 16-seed evals + margin 2.0.
        improved = (wr > best["wr"] + 1e-9) or \
                   (abs(wr - best["wr"]) < 1e-9 and rank <= best["rank"] - 2.0)
        if improved:
            ts = save_checkpoint(net, wr, rank)
            best = {"ts": ts, "wr": wr, "rank": rank, "source": "trainer"}
            msg += f" -> NEW BEST ts={ts}"
            if wr >= ship_wr:
                msg += f" | SHIP GATE (>= {ship_wr}) REACHED"
        log(msg)
        _hb("iter_done", t_it, loss=round(float(loss), 4),
            wr=round(float(wr), 3), rank=round(float(rank), 2),
            shards=len(pending_v2))
        # v2 shards stay ON HF (teacher needs them); only local move to done
        for p in pending_v2:
            p.rename(DONE_V2 / p.name)
    # shift-end safety: don't lose in-memory progress when the 9h kernel dies
    if net is not None:
        wr, rank = T.evaluate(net, seeds=int(os.environ.get("EVAL_SEEDS", "8")),
                              silent=True)
        if (wr > best["wr"] + 1e-9) or \
           (abs(wr - best["wr"]) < 1e-9 and rank <= best["rank"] - 1.5):
            ts = save_checkpoint(net, wr, rank)
            log(f"shift-end save: wr={wr:.2f} rank={rank:.2f} ts={ts}")
    log("trainer hours exhausted — exiting")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["worker", "trainer"])
    ap.add_argument("--hours", type=float, default=8)
    ap.add_argument("--skill", default="medium")
    ap.add_argument("--n-bots", type=int, default=8)
    a = ap.parse_args()
    log(f"LOOP {a.mode} hours={a.hours} device={T.DEVICE}")
    if a.mode == "worker":
        if os.environ.get("V2") == "1":
            mode_worker_v2(a.hours)
        else:
            mode_worker(a.hours, a.skill, a.n_bots)
    else:
        mode_trainer(a.hours)




# ================= v2 pipeline (128px + diff + numbers) =====================
SHARDS_V2 = RL / "shards_v2"
DONE_V2 = RL / "done_v2"
for d in (SHARDS_V2, DONE_V2):
    d.mkdir(parents=True, exist_ok=True)


def _cell_of(act, game=None, grid=32):
    if act.kind == "bank" or act.x is None:
        return 0
    w = game.w if game is not None else 400
    cx = min(grid - 1, max(0, int(act.x / w * grid)))
    cy = min(grid - 1, max(0, int(act.y / w * grid)))
    return cy * grid + cx


def pack_v2(rec):
    return dict(
        rgb=np.stack(rec["rgb"]).astype(np.uint8),
        lab=np.stack(rec["lab"]).astype(np.uint8),
        nums=np.stack(rec["nums"]).astype(np.float32),
        kind=np.array(rec["kind"], dtype=np.int64),
        cell=np.array(rec["cell"], dtype=np.int64),
        pct=np.array(rec["pct"], dtype=np.float32),
        logp=np.array(rec["logp"], dtype=np.float32),
        reward=np.array(rec["reward"], dtype=np.float32),
        alive=np.int64(rec["alive"]),
    )


def _true_logp(net, act, game):
    """Log-prob of the TAKEN action under the current net (needed for real
    PPO later; the old -2.0 placeholder made v2 shards unusable for PPO)."""
    dev = next(net.parameters()).device
    rgb, _ = game.frame_tensor(1, size=64)
    x = torch.tensor(rgb.transpose(2, 0, 1)[None], dtype=torch.float32,
                     device=dev) / 255.0
    with torch.no_grad():
        click, kind, _p, _v = net.forward(x, None)
        pk = T.F.softmax(kind[0], dim=-1)
        ki = {"expand": 0, "attack": 1, "bank": 2}[act.kind]
        lp = float(torch.log(pk[ki] + 1e-9))
        if ki != 2 and act.x is not None:
            cx = min(15, max(0, int(act.x / game.w * 16)))
            cy = min(15, max(0, int(act.y / game.h * 16)))
            pc = T.F.softmax(click[0], dim=-1)
            lp += float(torch.log(pc[cy * 16 + cx] + 1e-9))
    return lp


def unpack_v2_episodes(shard):
    """HD shards -> farmer-PPO episodes: rgb 256->64 (4x4 mean pool),
    nums[8] -> ctx[3] (me_frac, e1_frac, red), cell 32x32 -> 16x16,
    arena-map episodes dropped."""
    from audit_data import arena_eps_of
    eps = []
    rgb = shard["rgb"]
    n, h, w, c = rgb.shape
    rgb64 = (rgb.reshape(n, 64, 4, 64, 4, 3).mean(axis=(2, 4))
             .astype(np.float32) / 255.0).transpose(0, 3, 1, 2)  # CHW like v1
    nums = shard["nums"]
    ctx = nums[:, [1, 3, 2]].astype(np.float32)
    cell32 = shard["cell"]
    cy, cx = cell32 // 32, cell32 % 32
    cell16 = (cy // 2) * 16 + (cx // 2)
    ae = arena_eps_of(shard["lab"], shard["lens"])
    rec = [max(1, int(l) // 2) for l in shard["lens"]]
    off_r = 0
    off_s = 0
    for i, L in enumerate(shard["lens"]):
        r = rec[i]
        if not ae[i] and r >= 3:
            # obs were recorded every 2nd tick; rewards every tick.
            # Upsample obs x2 (last obs reused) so lengths align exactly.
            rep = lambda a: a[np.minimum(np.arange(L) // 2, a.shape[0] - 1)]
            eps.append({
                "rgb": rep(rgb64[off_r:off_r + r]),
                "ctx": rep(ctx[off_r:off_r + r]),
                "kind": rep(shard["kind"][off_r:off_r + r]),
                "cell": rep(cell16[off_r:off_r + r]),
                "pct": rep(shard["pct"][off_r:off_r + r]),
                "logp": rep(shard["logp"][off_r:off_r + r]),
                "reward": shard["reward"][off_s:off_s + L],
                "alive": bool(shard["alive"][i]),
            })
        off_r += r
        off_s += L
    return eps


def mode_worker_v2(hours):
    """Farmer for the 100M teacher: records 128px bundles + labels + numbers.
    Actions come from the current best net (any size that fits _policy_action
    via 64px); the BUNDLE is what the teacher learns from."""
    # BUGFIX 2026-08-12: train_nn seeds np.random(2026) at import, so every
    # worker drew the SAME map sequence (slot 0 = black_arena forever).
    # Reseed per process for real map variety.
    np.random.seed((os.getpid() * 2654435761 + int(time.time())) % 2**31)
    api0, _tok0 = _hf_api()   # used by the flush-failure budget guard below
    t_end = time.time() + hours * 3600
    ep_per = int(os.environ.get("V2_EP", "4"))
    flush_s = float(os.environ.get("FLUSH_S", "1200"))
    V2_SIZE = int(os.environ.get("V2_SIZE", "256"))     # HD eyes (was 128)
    rec_every = int(os.environ.get("REC_EVERY", "2"))   # record every 2nd tick
    GRID = V2_SIZE // 8
    last_flush = time.time()
    shard_i = 0
    while time.time() < t_end:
        net, meta = load_latest_net()
        net.eval()
        sk = str(np.random.choice(["medium"] * 3 + ["easy", "hard"]))
        nb = int(np.random.choice([8, 12, 16, 24, 32]))  # chaotic but survivable
        seed0 = int(time.time()) % 100000
        episodes = []
        for e in range(ep_per):
            game = T._make_game(sk, seed0 + e * 13 + 1, n_bots=nb)
            rec = {k: [] for k in ("rgb", "lab", "nums", "kind", "cell",
                                   "pct", "logp", "reward")}
            rec["alive"] = False
            done = False
            while game.tick < game.max_ticks and not done:
                st = game.state_for(1)
                if not st.self_blob:
                    break
                act, pctv = T._policy_action(net, st, game)
                if game.tick % rec_every == 0:
                    rgb_s, lab_s, nums = game.frame_bundle(1, V2_SIZE)
                    rec["rgb"].append(rgb_s)
                    rec["lab"].append(lab_s)
                    rec["nums"].append(nums)
                    rec["kind"].append({"expand": 0, "attack": 1, "bank": 2}[act.kind])
                    rec["cell"].append(_cell_of(act, game, GRID))
                    rec["pct"].append(float(pctv))
                    rec["logp"].append(_true_logp(net, act, game))
                area_before = int((game.world == 1).sum())
                kills_before = game.players[1].kills
                actions = {1: game._clicks_for(act, T.SIM["clicks_per_tick"])}
                for pid in game._pids:
                    if pid == 1 or not game.players[pid].alive:
                        continue
                    actions[pid] = game._bot_clicks(pid)
                game.step(actions)
                area_after = int((game.world == 1).sum())
                reward = (area_after - area_before) / 2000.0
                if game.players[1].kills > kills_before:
                    reward += 2.0
                if game.players[1].alive:
                    reward += 0.005
                    if area_after <= area_before:
                        reward -= 0.002
                else:
                    reward -= 1.0
                    done = True
                    rec["reward"].append(reward)
                    break
                rec["reward"].append(reward)
                if len(rec["reward"]) >= 250:
                    break
            rec["alive"] = bool(game.players[1].alive)
            # quality filter (user-approved 2026-08-09): degenerate episodes
            # (instant deaths / no real play) are noise, not data
            if len(rec["reward"]) >= 10:
                episodes.append(rec)
        if not episodes:
            time.sleep(10)
            continue
        path = SHARDS_V2 / f"shard_v2_{int(time.time())}_{os.getpid()}_{shard_i}.npz"
        merged = {k: np.concatenate([pack_v2(ep)[k] for ep in episodes])
                  for k in ("rgb", "lab", "nums", "kind", "cell", "pct",
                            "logp", "reward")}
        merged["alive"] = np.array([pack_v2(ep)["alive"] for ep in episodes])
        merged["lens"] = np.array([len(ep["reward"]) for ep in episodes])
        np.savez_compressed(path, **merged)
        shard_i += 1
        log(f"v2 shard {path.name}: {len(episodes)} eps "
            f"({merged['rgb'].shape[0]} steps)")
        if time.time() - last_flush > flush_s:
            if push_shards_batch(SHARDS_V2):
                flush_s = 900.0
            else:
                flush_s = min(flush_s * 2, 7200.0)   # quiet protocol
                # budget guard only when HF is actually reachable
                loc = sorted(SHARDS_V2.glob("shard_v2_*.npz"))
                if api0 and len(loc) > 2:
                    for f in loc[:-2]:
                        f.unlink()
            last_flush = time.time()
    push_shards_batch(SHARDS_V2)
    log("v2 worker done")


if __name__ == "__main__":
    main()
