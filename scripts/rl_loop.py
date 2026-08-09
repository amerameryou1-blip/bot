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


def _hb(phase, t_it=None, **extra):
    """v6 telemetry: heartbeat with PHASE per sub-step, 3 upload attempts
    with backoff. If the trainer ever hangs, the LAST phase tells where."""
    payload = {"ts": int(time.time()), "phase": phase}
    if t_it is not None:
        payload["iter_s"] = int(time.time() - t_it)
    payload.update(extra)
    for attempt in range(3):
        try:
            api_h, tok_h = _hf_api()
            if not api_h:
                return
            hb = RL / "hb.json"
            json.dump(payload, open(hb, "w"))
            api_h.upload_file(path_or_fileobj=str(hb),
                              path_in_repo="rl/trainer_heartbeat.json",
                              repo_id=HF_DATASET, repo_type="dataset", token=tok_h)
            return
        except Exception:
            time.sleep(15 * (attempt + 1))
    log(f"[hb] phase={phase} upload failed 3x")


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


def load_latest_net():
    """Newest checkpoint: HF rl/best.json+ckpt -> local best.pt -> model.pt.
    All loads are shape-safe (old 85k brains can never crash the 2M net)."""
    api, tok = _hf_api()
    if api:
        try:
            remote = json.loads(api.hf_hub_download(
                repo_id=HF_DATASET, repo_type="dataset",
                filename="rl/best.json", token=tok))
            local = best_meta()
            if remote.get("ts", 0) > local.get("ts", 0):
                p = api.hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                                        filename=f"rl/ckpt_{remote['ts']}.pt", token=tok)
                net = T.make_net().to(T.DEVICE)
                if _try_load(net, p):
                    log(f"pulled newer checkpoint ts={remote['ts']} "
                        f"wr={remote.get('wr')}")
                    return net, remote
        except Exception as e:
            log(f"checkpoint pull skipped ({str(e)[:60]})")
    net = T.make_net().to(T.DEVICE)
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


def push_shards_batch(folder: Path):
    """ONE commit for all pending local shards (HF caps commits at
    128/h — per-shard commits were saturating the account budget)."""
    files = sorted(folder.glob("shard_rl_*.npz"))
    if not files:
        return
    api, tok = _hf_api()
    if not api:
        return
    try:
        api.upload_folder(folder_path=str(folder), path_in_repo="rl/shards",
                          repo_id=HF_DATASET, repo_type="dataset", token=tok,
                          ignore_patterns="*.partial")
        log(f"{len(files)} shards -> HF (1 commit)")
        for f in files:
            f.unlink()
    except Exception as e:
        log(f"batch upload failed ({str(e)[:60]}) — kept local")


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
        nb = int(np.random.choice([6, 8, 10, 12]))
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
            push_shards_batch(SHARDS)   # 1 HF commit per flush
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
    # seed best wr from the loaded checkpoint if we have one
    if best["wr"] < 0 and T.MODEL_PT.exists():
        net = T.make_net().to(T.DEVICE)
        T.load_model(net)
        wr, rank = T.evaluate(net, seeds=6, silent=True)
        best = {"ts": int(time.time()), "wr": wr, "rank": rank, "source": "v14-seed"}
        json.dump(best, open(BEST_JSON, "w"), indent=2)
        log(f"seeded best from v14 checkpoint: wr={wr:.2f} rank={rank:.2f}")
    while time.time() < t_end:
        t_it = time.time()
        # heartbeat FIRST, before any heavy work (v6)
        _hb("iter_start", t_it)
        log(f"[hb] iter start (pending local={len(list(SHARDS.glob('shard_*.npz')))})")
        if api:
            try:
                files = api.list_repo_files(HF_DATASET, repo_type="dataset", token=tok)
                new = [f for f in files
                       if f.startswith("rl/shards/shard_rl_") and f.endswith(".npz")
                       and not (SHARDS / Path(f).name).exists()
                       and not (DONE / Path(f).name).exists()]
                for f in new[:8]:
                    p = api.hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                                            filename=f, token=tok)
                    # shutil.move: os.replace fails cross-device on Kaggle
                    # (cache on /root, working dir on /kaggle) — the bug that
                    # spun the trainer idle for hours (found 2026-08-09)
                    shutil.move(p, str(SHARDS / Path(f).name))
                if new:
                    log(f"pulled {len(new)} new shards from HF")
            except Exception as e:
                log(f"shard pull skipped ({str(e)[:60]})")
        pending = sorted(SHARDS.glob("shard_*.npz"))
        if not pending:
            time.sleep(30)
            continue
        episodes = []
        for p in pending:
            try:
                episodes += unpack_episodes(np.load(p))
            except Exception as e:
                log(f"bad shard {p.name}: {e}")
        if not episodes:
            for p in pending:
                p.rename(DONE / p.name)
            continue
        # cap the PPO batch: 2M-param PPO is heavy on CPU; more iterations
        # with fresher data beats giant slow ones (measured 2026-08-09)
        cap = int(os.environ.get("EP_CAP", "24"))
        if len(episodes) > cap:
            idx = np.random.choice(len(episodes), cap, replace=False)
            episodes = [episodes[i] for i in sorted(idx)]
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
        msg = (f"trainer iter: shards={len(pending)} episodes={len(episodes)} "
               f"loss={loss:.3f} | LAST-SURVIVOR wr={wr:.2f} rank={rank:.2f} "
               f"(best {best['wr']:.2f})")
        if wr >= best["wr"]:
            ts = save_checkpoint(net, wr, rank)
            best = {"ts": ts, "wr": wr, "rank": rank, "source": "trainer"}
            msg += f" -> NEW BEST ts={ts}"
            if wr >= ship_wr:
                msg += f" | SHIP GATE (>= {ship_wr}) REACHED"
        log(msg)
        _hb("iter_done", t_it, loss=round(float(loss), 4),
            wr=round(float(wr), 3), rank=round(float(rank), 2),
            shards=len(pending))
        consumed = [p.name for p in pending]
        for p in pending:
            p.rename(DONE / p.name)
        # storage hygiene (user 2026-08-08): experience is now IN the weights —
        # delete consumed shards from HF in ONE commit so the repo stays lean.
        # (DeleteFiles was renamed across hub versions — detect defensively.)
        try:
            api_del, tok_del = _hf_api()
            if api_del:
                import huggingface_hub as _h
                op = None
                for nm in ("DeleteFiles", "CommitOperationDelete",
                           "CommitOperationDeleteFile"):
                    if hasattr(_h, nm):
                        op = getattr(_h, nm)
                        break
                if op:
                    api_del.create_commit(
                        repo_id=HF_DATASET, repo_type="dataset", token=tok_del,
                        commit_message="consume shards",
                        operations=[op(path_in_repo=f"rl/shards/{n}")
                                    for n in consumed])
        except Exception as e:
            log(f"shard delete commit failed ({str(e)[:60]})")
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
        mode_worker(a.hours, a.skill, a.n_bots)
    else:
        mode_trainer(a.hours)


if __name__ == "__main__":
    main()
