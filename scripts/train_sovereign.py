#!/usr/bin/env python3
"""
train_sovereign.py — Stage-A trainer for SOVEREIGN v3 (~290M dense),
the rival's architecture adopted 2026-08-14 by user decision, with the
pipeline's proven plumbing bolted on.

GOOD THINGS TAKEN FROM THE OLD TEACHER PIPELINE:
  * strict arena gating    — audit_data.arena_eps_of (measured lobby-
                             screenshot signatures), wired into the adapter
  * AWR frame weights      — survivor x4, kill-window x2 (sovereign_data)
  * return conditioning    — RTG + intent mirrors + world model (his design)
  * honest eval            — 16-seed last-survivor semantics copied from
                             train_nn.evaluate ("win" = ONLY player alive)
  * strict publish gate    — wr gain, or rank-2.0 at the wr=0 plateau;
                             no tie-floods (2026-08-14 lesson)
  * shift-end safety       — final ckpt saved/uploaded on exit
  * separate namespace     — rl/sovereign/{best.json,ckpt_*.pt} so the
                             teacher lineage (rl/ckpt_*.pt) is untouched

LOCAL SMOKE (2GB sandbox):
  python3 scripts/train_sovereign.py --mini --local-shard <shard.npz> \
      --steps 3 --eval-seeds 1 --no-hf

FULL TRAINING (Kaggle T4x2, ONLY after the 15GB gate + /tmp/review_ok):
  python3 scripts/train_sovereign.py --device cuda --amp --bs 16 \
      --epochs 2 --fetch 400

GPU budget note: this script waits for the gate like everything else.
CPU quota stays with the data fleet; do not run this on Kaggle CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nn.sovereign import Sovereign, SovereignConfig, stage_a_loss, count_params  # noqa: E402
from sovereign_data import ShardPrepper, StageADataset  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

HF_REPO = "amer224/territorial-bot-data"
# Kaggle's /tmp may be small; kernel sets SOV_TMP=/kaggle/working
TMP = Path(os.environ.get("SOV_TMP", "/tmp"))
SOV_DIR = "rl/sovereign"

WM_LAMB = dict(z=0.5, r=0.1, prior=0.3, v=0.1, w=0.1)


# --------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------

def mini_config() -> SovereignConfig:
    """Thin SOVEREIGN at real 256px — fits a 2GB sandbox for smoke tests.
    Same spatial contract (gate/seg heads sit at H/4 = 64x64 == lab64)."""
    return SovereignConfig(
        map_size=256,
        enc_ch=(24, 24, 32, 48, 64, 80), enc_blocks=(1, 1, 1, 1, 1),
        cortex_d=128, cortex_heads=4, cortex_ffn=256, cortex_layers=1,
        dec_plan=((128, 64, 1), (64, 48, 1), (48, 32, 1),
                  (32, 24, 1), (16, None, 1), (8, None, 1)),
        dec_out_ch=8, heat_hidden=16, tower_hidden=16,
        head_hidden=64, pct_hidden=64, mem_hidden=32,
        lat_c=8, dyn_hidden=64, act_embed=64)


def get_shard_paths(args) -> list:
    paths = []
    if args.local_shard:
        paths = [Path(args.local_shard)]
    elif args.data_dir:
        paths = sorted(Path(args.data_dir).glob("shard_v2_*.npz"))
    if args.fetch and args.fetch > 0:
        try:
            from huggingface_hub import HfApi, hf_hub_download
            tok = os.environ.get("HF_TOKEN", "")
            api = HfApi(token=tok or None)
            fs = [f for f in api.list_repo_tree(
                      HF_REPO, path_in_repo="rl/shards_v2",
                      repo_type="dataset", token=tok or None)
                  if getattr(f, "size", 0) > 0]
            fs.sort(key=lambda f: f.rfilename)
            fs = fs[-args.fetch:]
            dl = Path(args.dl_dir)
            for f in fs:
                dst = dl / Path(f.rfilename).name
                if not dst.exists():
                    hf_hub_download(HF_REPO, f.rfilename,
                                    repo_type="dataset", token=tok or None,
                                    local_dir=str(dl))
            paths += sorted(dl.glob("rl/shards_v2/shard_v2_*.npz"))
        except Exception as e:
            print(f"[warn] HF fetch failed: {e}", flush=True)
    seen, out = set(), []
    for p in paths:
        p = Path(p)
        if p.exists() and str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


# --------------------------------------------------------------------------
# world-model loss (train_notes.md A.3, MuZero-lite)
# --------------------------------------------------------------------------

def _f32(o):
    # 2026-08-15: losses run OUTSIDE autocast in fp32 (BCE is autocast-unsafe)
    if isinstance(o, tuple):
        return tuple(_f32(x) for x in o)
    if torch.is_tensor(o):
        return o.float()
    return o


def world_model_loss(model, batch, policy_out, device, amp_on=False):
    rgb = batch["rgb"].to(device)
    nums = batch["nums"].to(device)
    rtg = batch["rtg"].to(device).view(-1)
    with torch.autocast("cuda", enabled=amp_on):
        z = model.encode(rgb, nums, rtg=rtg)
        prior, v_hat, w_hat = model.prediction(z)
        z_next_hat, r_hat = model.dynamics(
            z, batch["kind"].to(device), batch["cell"].to(device),
            batch["pct"].to(device))
        with torch.no_grad():
            z_next = model.encode(batch["rgb_next"].to(device),
                                  batch["nums"].to(device),
                                  rtg=batch["rtg_next"].to(device).view(-1))
    z, prior, v_hat, w_hat = _f32((z, prior, v_hat, w_hat))
    z_next_hat, r_hat, z_next = _f32((z_next_hat, r_hat, z_next))
    l_z = F.mse_loss(z_next_hat, z_next)
    l_r = F.huber_loss(r_hat, batch["reward"].to(device))
    joint = policy_out["cell_logits"].detach().flatten(1)   # (B, 3*g*g)
    joint_p = F.softmax(joint, dim=1)
    l_prior = F.kl_div(F.log_softmax(prior, dim=1), joint_p,
                       reduction="batchmean")
    l_v = F.huber_loss(v_hat, batch["ret"].to(device))
    l_w = F.cross_entropy(w_hat, batch["win_lab"].to(device))
    total = (WM_LAMB["z"] * l_z + WM_LAMB["r"] * l_r
             + WM_LAMB["prior"] * l_prior + WM_LAMB["v"] * l_v
             + WM_LAMB["w"] * l_w)
    return total, dict(wm_z=l_z.item(), wm_r=l_r.item(),
                       wm_prior=l_prior.item())


# --------------------------------------------------------------------------
# honest last-survivor eval (copy of train_nn.evaluate semantics, 256px)
# --------------------------------------------------------------------------

def evaluate_sovereign(net, seeds=16, silent=True, rtg_asp=0.0):
    import train_nn as T
    from bot.planner import ClickAction
    net.eval()
    dev = next(net.parameters()).device
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
            rgb, _ = game.frame_tensor(1, size=256)
            x = torch.tensor(rgb.transpose(2, 0, 1)[None],
                             dtype=torch.float32, device=dev) / 255.0
            total = float(game.h * game.w)
            me_frac = st.self_blob.area / total
            enemy_frac = max((e.area for e in st.enemies), default=0) / total
            nums = torch.tensor([[0.0, me_frac, enemy_frac, 0.0, 0.0,
                                  game.tick / max(game.max_ticks, 1),
                                  0.0, 0.0]], dtype=torch.float32, device=dev)
            with torch.no_grad():
                a = net.act(x, nums, rtg=torch.tensor([rtg_asp], device=dev))
            kind_i = int(a["kind"][0])
            pctv = float(a["pct"][0])
            if kind_i == 2:
                act = ClickAction("bank", reason="sov-bank")
            else:
                cy = int(a["yx"][0][0]); cx = int(a["yx"][0][1])
                y = (cy + 0.5) / 256.0 * game.h
                xp = (cx + 0.5) / 256.0 * game.w
                kind_s = {0: "expand", 1: "attack"}[kind_i]
                act = ClickAction(kind_s, float(xp), float(y),
                                  pctv * 100.0 if kind_i == 1 else 0.0,
                                  reason=f"sov-{kind_s}")
            actions = {1: game._clicks_for(act, T.SIM["clicks_per_tick"])}
            for pid in game._pids:
                if pid == 1 or not game.players[pid].alive:
                    continue
                actions[pid] = game._bot_clicks(pid)
            game.step(actions)
        alive = game.players[1].alive and (game.world == 1).sum() > 0
        alive_list = [pid for pid in game._pids if game.players[pid].alive]
        is_last = bool(alive) and len(alive_list) == 1   # HONEST gate
        wins += 1 if is_last else 0
        survived += 1 if alive else 0
        areas = {pid: int((game.world == pid).sum()) for pid in game._pids}
        rank = 1 + sum(1 for pid, ar in areas.items()
                       if pid != 1 and ar > areas[1])
        ranks.append(rank)
    wr = wins / seeds
    avg_rank = sum(ranks) / len(ranks)
    if not silent:
        print(f"EVAL-SOV ({T.SIM['n_bots']}-player): wr={wr:.2f} "
              f"alive={survived / seeds:.2f} rank={avg_rank:.2f}", flush=True)
    return wr, avg_rank


# --------------------------------------------------------------------------
# HF publish (separate namespace; strict gate; validated best.json)
# --------------------------------------------------------------------------

def _hf_token():
    return os.environ.get("HF_TOKEN", "")


def load_best() -> dict:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(HF_REPO, f"{SOV_DIR}/best.json",
                            repo_type="dataset", token=_hf_token() or None,
                            local_dir=str(TMP / "best"))
        d = json.load(open(p))
        if isinstance(d.get("rank"), (int, float)):
            return {"ts": int(d.get("ts", 0)), "wr": float(d.get("wr", 0)),
                    "rank": float(d["rank"]), "source": "sovereign"}
    except Exception:
        pass
    return {"ts": 0, "wr": 0.0, "rank": 99.0, "source": "sovereign"}


def hf_upload(local: Path, remote: str):
    from huggingface_hub import upload_file
    upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                repo_id=HF_REPO, repo_type="dataset",
                token=_hf_token() or None)


def heartbeat(**kw):
    kw["ts"] = time.time()
    # 2026-08-15: two redundant GPU kernels raced on ONE heartbeat path and
    # the HF copy read back EMPTY. Per-kernel filename kills the race.
    slug = os.environ.get("KAGGLE_KERNEL_SLUG", "local")
    p = TMP / f"sov_heartbeat_{slug}.json"
    json.dump(kw, open(p, "w"))
    if _hf_token() and not kw.get("no_hf"):
        try:
            hf_upload(p, f"rl/sovereign_heartbeat_{slug}.json")
        except Exception as e:
            print("[warn] hb upload:", str(e)[:80], flush=True)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true",
                    help="thin config for sandbox smoke tests")
    ap.add_argument("--local-shard", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--fetch", type=int, default=0,
                    help="pull N newest shards from HF")
    ap.add_argument("--dl-dir", default=str(__import__("os").environ.get("SOV_TMP", "/tmp") + "/sov_shards"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=0,
                    help="cap optimizer steps (0 = no cap)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-seeds", type=int, default=16)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--no-hf", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available()
                       or args.device == "cpu" else "cpu")
    cfg = mini_config() if args.mini else SovereignConfig()
    cfg.grad_ckpt = args.grad_ckpt
    model = Sovereign(cfg).to(dev)
    print(f"[sov] params={count_params(model) / 1e6:.2f}M device={dev} "
          f"mini={args.mini}", flush=True)
    if args.resume and Path(args.resume).exists():
        model.load_state_dict(torch.load(args.resume, map_location=dev))
        print(f"[sov] resumed {args.resume}", flush=True)

    paths = get_shard_paths(args)
    print(f"[sov] {len(paths)} shards available", flush=True)
    if not paths:
        print("[sov] no shards — exiting"); return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev.type == "cuda")
    total_steps = (args.steps if args.steps > 0 else
                   max(1, len(paths) * 40 * args.epochs))
    step = 0
    best = load_best() if not args.no_hf else {"ts": 0, "wr": 0.0,
                                                "rank": 99.0,
                                                "source": "sovereign"}
    print(f"[sov] best so far: wr={best['wr']:.2f} rank={best['rank']:.2f}",
          flush=True)
    t0 = time.time()
    last_hb = 0.0
    running = {}

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / max(args.warmup, 1)
        frac = (s - args.warmup) / max(total_steps - args.warmup, 1)
        return 3e-5 + 0.5 * (args.lr - 3e-5) * (1 + math.cos(math.pi * min(frac, 1.0)))

    def do_eval(tag):
        nonlocal best
        seeds = args.eval_seeds
        wr, rank = evaluate_sovereign(model, seeds=seeds)
        print(f"[sov] EVAL({tag}, {seeds} seeds): wr={wr:.2f} "
              f"rank={rank:.2f}", flush=True)
        improved = (wr > best["wr"] + 1e-9) or \
                   (abs(wr - best["wr"]) < 1e-9 and rank <= best["rank"] - 2.0)
        if improved:
            ts = int(time.time())
            ck = TMP / f"sov_ckpt_{ts}.pt"
            torch.save(model.state_dict(), ck)
            best = {"ts": ts, "wr": wr, "rank": rank, "source": "sovereign"}
            if not args.no_hf and _hf_token():
                try:
                    hf_upload(ck, f"{SOV_DIR}/ckpt_{ts}.pt")
                    bp = TMP / "sov_best.json"; json.dump(best, open(bp, "w"))
                    hf_upload(bp, f"{SOV_DIR}/best.json")
                    print(f"[sov] NEW BEST uploaded ts={ts}", flush=True)
                except Exception as e:
                    print("[warn] ckpt upload:", str(e)[:120], flush=True)
            else:
                print(f"[sov] NEW BEST (local only) ts={ts}", flush=True)

    stop_at = t0 + args.max_minutes * 60 if args.max_minutes > 0 else None
    try:
        for epoch in range(args.epochs):
            order = paths.copy()
            np.random.shuffle(order)
            for sp in order:
                if stop_at and time.time() > stop_at:
                    raise KeyboardInterrupt("max-minutes")
                try:
                    prepped = ShardPrepper(sp).prepare()
                except Exception as e:
                    print(f"[sov] prep fail {sp.name}: {str(e)[:80]}",
                          flush=True)
                    continue
                if prepped is None or len(prepped.rgb) < 8:
                    continue
                ds = StageADataset([prepped])
                sampler = WeightedRandomSampler(
                    ds.w, num_samples=max(len(ds) // 2, args.bs),
                    replacement=True)
                dl = DataLoader(ds, batch_size=args.bs, sampler=sampler,
                                num_workers=0, drop_last=True)
                for batch in dl:
                    if args.steps and step >= args.steps:
                        raise KeyboardInterrupt("steps cap")
                    if stop_at and time.time() > stop_at:
                        raise KeyboardInterrupt("max-minutes")
                    for g in opt.param_groups:
                        g["lr"] = lr_at(step)
                    rgb = batch["rgb"].to(dev)
                    nums = batch["nums"].to(dev)
                    rtg = batch["rtg"].to(dev).view(-1)
                    cell = batch["cell"].to(dev)
                    # 2026-08-15: forward under autocast, losses in fp32
                    # (F.binary_cross_entropy is autocast-unsafe -> v2 crash)
                    amp_on = args.amp and dev.type == "cuda"
                    with torch.autocast("cuda", enabled=amp_on):
                        out = model(rgb, nums, rtg=rtg, cell=cell,
                                    return_all=True)
                    out = {k: _f32(v) for k, v in out.items()}
                    loss, terms = stage_a_loss(
                        out, batch["kind"].to(dev), cell,
                        batch["pct"].to(dev), ret=batch["ret"].to(dev),
                        win_lab=batch["win_lab"].to(dev),
                        lab64=batch["lab64"].to(dev),
                        lab64_next=batch["lab64_next"].to(dev),
                        gate_mask=batch["gate_mask"].to(dev),
                        gate_valid=batch["gate_valid"].to(dev),
                        threat=batch["threat"].to(dev),
                        expand=batch["expand"].to(dev),
                        nums_next=batch["nums_next"].to(dev),
                        w=batch["w"].to(dev))
                    wm, wm_t = world_model_loss(model, batch, out, dev,
                                                amp_on)
                    total = loss + wm
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(total).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                    step += 1
                    running = {k: float(terms.get(k, 0)) for k in
                               ("kind_nll", "cell_nll", "pct_nll")}
                    running.update(wm_t)
                    if step % 25 == 0:
                        mem = (torch.cuda.max_memory_allocated() / 1e9
                               if dev.type == "cuda" else 0.0)
                        print(f"[sov] step={step} total={float(total):.3f} "
                              f"gpu_peak={mem:.1f}GB "
                              f"kind={running.get('kind_nll', 0):.3f} "
                              f"cell={running.get('cell_nll', 0):.3f} "
                              f"wm_z={running.get('wm_z', 0):.4f}", flush=True)
                    if step % args.eval_every == 0:
                        # 2026-08-15: runs 1+2 both died around first eval.
                        # A mid-run eval crash must NEVER kill training.
                        try:
                            do_eval(f"step{step}")
                        except Exception as e:
                            print(f"[warn] mid-eval failed: {str(e)[:150]}",
                                  flush=True)
                    if time.time() - last_hb > 120:
                        last_hb = time.time()
                        heartbeat(phase="training", step=step,
                                  loss=float(total.detach()), no_hf=args.no_hf,
                                  **{k: round(v, 4) for k, v in running.items()})
        do_eval("final")
    except KeyboardInterrupt as e:
        print(f"[sov] stop: {e}", flush=True)
    # shift-end safety (old-teacher lesson): save where we are.
    # 2026-08-15 fix: run1 uploaded the final ckpt but DIED in the unguarded
    # eval below it -> checkpoint with zero eval numbers. Save + heartbeat
    # FIRST, eval guarded.
    ck = TMP / "sov_final.pt"
    torch.save(model.state_dict(), ck)
    if not args.no_hf and _hf_token():
        try:
            hf_upload(ck, f"{SOV_DIR}/ckpt_final_{int(time.time())}.pt")
        except Exception as e:
            print("[warn] final upload:", str(e)[:120], flush=True)
    heartbeat(phase="final_ckpt_saved", step=step, no_hf=args.no_hf)
    try:
        wr, rank = evaluate_sovereign(model, seeds=max(args.eval_seeds, 1))
        print(f"[sov] END-EVAL: wr={wr:.2f} rank={rank:.2f} steps={step}",
              flush=True)
        if (wr > best["wr"] + 1e-9) or \
           (abs(wr - best["wr"]) < 1e-9 and rank <= best["rank"] - 2.0):
            ts = int(time.time())
            best = {"ts": ts, "wr": wr, "rank": rank, "source": "sovereign"}
            bp = TMP / "sov_best.json"
            json.dump(best, open(bp, "w"))
            if not args.no_hf and _hf_token():
                hf_upload(bp, f"{SOV_DIR}/best.json")
                print(f"[sov] BEST uploaded ts={ts}", flush=True)
    except Exception as e:
        print("[warn] final eval failed:", str(e)[:150], flush=True)
    heartbeat(phase="done", step=step, no_hf=args.no_hf)
    print("[sov] DONE", flush=True)


if __name__ == "__main__":
    main()
