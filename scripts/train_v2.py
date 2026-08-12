#!/usr/bin/env python3
"""v2 training pipeline (100M teacher / 9.3M student), GPU-ready.

Stages:
  sup      — supervised pretrain TEACHER on v2 shards:
             seg (labels come free from the sim) + click-clone + kind + pct
  distill  — TEACHER -> STUDENT (seg/click/kind logits KL + hard labels)
  ppo      — PPO for the teacher on v2 rollouts (GPU; later)
  eval     — last-survivor win-rate of a net (student by default)

Data: rl/shards_v2/*.npz on HF (128px rgb + labels + nums + actions).
Diff channel = rgb[t]-rgb[t-1] built at load time (motion eyes).
"""
import os
import sys
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F

from nn.model_v2 import TeacherV2, StudentV2, count_params
import rl_loop as R
import train_nn as T

DEVICE = T.DEVICE
W2 = T.WEIGHTS / "v2"
W2.mkdir(parents=True, exist_ok=True)
TEACH_PT = W2 / "teacher.pt"
STUD_PT = W2 / "student.pt"
GRID = 16


def _list_shards():
    return sorted(R.SHARDS_V2.glob("shard_v2_*.npz"))


def pull_shards():
    api, tok = R._hf_api()
    if not api:
        return
    try:
        import shutil
        files = api.list_repo_files(R.HF_DATASET, repo_type="dataset", token=tok)
        new = [f for f in files
               if (f.startswith("rl/shards_v2/") or
                   (f.startswith("rl/shards/shard_v2") ))  # old workers misfoldered
               and not (R.SHARDS_V2 / Path(f).name).exists()
               and not (R.DONE_V2 / Path(f).name).exists()]
        for f in new[:300]:
            p = api.hf_hub_download(R.HF_DATASET, f, repo_type="dataset", token=tok)
            shutil.move(p, str(R.SHARDS_V2 / Path(f).name))
        if new:
            print(f"pulled {len(new)} v2 shards", flush=True)
    except Exception as e:
        print(f"v2 pull skipped: {str(e)[:60]}", flush=True)


def _batches(shards, bs, epochs):
    for ep in range(epochs):
        np.random.shuffle(shards)
        for sp in shards:
            try:
                d = np.load(sp)
            except Exception as e:
                print("skip bad shard", sp.name, str(e)[:60])
                continue
            # arena-episode filter (2026-08-12): lobby-screenshot maps leaked
            # into the pool AND a seeding bug made every worker replay the same
            # map sequence (slot0=black_arena). Filter per EPISODE, keep the
            # good 75% of each shard. Signatures: black_arena water .92 +
            # land-thin .48; white_arena water .004; real watery maps thin .12.
            from audit_data import arena_eps_of
            lens = d["lens"]
            ae = arena_eps_of(d["lab"], lens)
            if all(ae):
                print("skip all-arena shard", sp.name)
                continue
            keep = np.concatenate([
                np.zeros(max(1, int(l) // 2), dtype=bool) if a
                else np.ones(max(1, int(l) // 2), dtype=bool)
                for l, a in zip(lens, ae)])
            rgb = d["rgb"]
            n = len(rgb)
            for i in range(0, n, bs):
                sel = np.nonzero(keep[i:i + bs])[0]
                if len(sel) < 4:
                    continue
                r = rgb[i:i + bs][sel].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                prev = np.concatenate([r[:1], r[:-1]])
                diff = np.clip(r - prev, -1, 1) * 0.5 + 0.5
                x = np.concatenate([r, diff], 1)
                lab = d["lab"][i:i + bs][sel]
                g = r.shape[2] // 8
                lab16 = torch.tensor(
                    _nn_resize(lab, g), dtype=torch.int64, device=DEVICE)
                yield (torch.tensor(x, device=DEVICE), lab16,
                       torch.tensor(d["nums"][i:i + bs][sel], device=DEVICE),
                       torch.tensor(d["cell"][i:i + bs][sel], device=DEVICE),
                       torch.tensor(d["kind"][i:i + bs][sel], device=DEVICE),
                       torch.tensor(d["pct"][i:i + bs][sel], device=DEVICE))


def _nn_resize(lab, size):
    import numpy as _np
    n, h, w = lab.shape
    ys = (np.arange(size) * h // size)
    xs = (np.arange(size) * w // size)
    return lab[:, ys][:, :, xs]


def stage_sup(epochs=2, bs=16, lr=3e-4):
    pull_shards()
    shards = _list_shards()
    if not shards:
        print("[sup] no v2 shards yet — skipping", flush=True)
        return None
    if os.environ.get("SMOKE") == "1":
        shards = shards[:1]
    net = TeacherV2().to(DEVICE)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)   # T4x2 ready
        print(f"[sup] DataParallel on {torch.cuda.device_count()} GPUs")
    print(f"[sup] teacher {count_params(net)/1e6:.1f}M on {len(shards)} shards",
          flush=True)
    opt = T.make_optimizer(net)
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for x, lab16, nums, cell, kind, pct in _batches(shards, bs, 1):
            seg, click, kd, pc, _ = net(x, nums, return_all=True)
            loss = F.cross_entropy(seg, lab16, label_smoothing=0.05)
            loss = loss + F.cross_entropy(click, cell) * 1.0
            loss = loss + F.cross_entropy(kd, kind) * 0.5
            m = kind == 1
            if m.any():
                loss = loss + F.mse_loss(pc[m], pct[m]) * 1.0
            if not opt.finite_ok(loss):
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if os.environ.get("SMOKE") == "1" and nb >= 3:
                break
        print(f"  sup epoch {ep+1}: loss={tot/max(nb,1):.4f}", flush=True)
        if os.environ.get("SMOKE") == "1":
            break
    sd = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    torch.save(sd, TEACH_PT)
    print(f"[sup] saved {TEACH_PT}", flush=True)
    return net


def stage_distill(epochs=2, bs=16):
    pull_shards()
    shards = _list_shards()
    if not shards or not TEACH_PT.exists():
        print("[distill] missing teacher/shards — skipping", flush=True)
        return None
    if os.environ.get("SMOKE") == "1":
        shards = shards[:1]
    teacher = TeacherV2().to(DEVICE)
    teacher.load_state_dict(torch.load(TEACH_PT, map_location=DEVICE))
    teacher.eval()
    stu = StudentV2().to(DEVICE)
    print(f"[distill] student {count_params(stu)/1e6:.1f}M", flush=True)
    opt = T.make_optimizer(stu)
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for x, lab16, nums, cell, kind, pct in _batches(shards, bs, 1):
            with torch.no_grad():
                tseg, tclick, tkd, _, _ = teacher(x, nums, return_all=True)
            seg, click, kd, pc, _ = stu(x, nums, return_all=True)
            loss = F.cross_entropy(seg, lab16, label_smoothing=0.05) * 0.5
            loss = loss + F.kl_div(F.log_softmax(seg, 1),
                                   F.softmax(tseg, 1), reduction="batchmean")
            loss = loss + F.kl_div(F.log_softmax(click, 1),
                                   F.softmax(tclick, 1), reduction="batchmean")
            loss = loss + F.kl_div(F.log_softmax(kd, 1),
                                   F.softmax(tkd, 1), reduction="batchmean")
            loss = loss + F.cross_entropy(click, cell) * 0.5
            if not opt.finite_ok(loss):
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if os.environ.get("SMOKE") == "1" and nb >= 3:
                break
        print(f"  distill epoch {ep+1}: loss={tot/max(nb,1):.4f}", flush=True)
        if os.environ.get("SMOKE") == "1":
            break
    torch.save(stu.state_dict(), STUD_PT)
    print(f"[distill] saved {STUD_PT}", flush=True)
    return stu


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "sup"
    if stage == "sup":
        stage_sup(epochs=int(os.environ.get("EPOCHS", "2")))
    elif stage == "distill":
        stage_distill(epochs=int(os.environ.get("EPOCHS", "2")))
    elif stage == "ppo":
        stage_ppo_v2(rounds=int(os.environ.get("ROUNDS", "8")))
    print("V2 DONE", flush=True)


if __name__ == "__main__":
    main()


# ============================== PPO for the 100M teacher ====================
def _bundle_x(game, prev, size=None):
    size = size or int(os.environ.get("V2_SIZE", "256"))
    rgb, lab, nums = game.frame_bundle(1, size)
    r = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    if prev is None:
        diff = np.full_like(r, 0.5)
    else:
        diff = np.clip(r - prev, -1, 1) * 0.5 + 0.5
    x = np.concatenate([r, diff], 1)
    return torch.tensor(x, device=DEVICE), torch.tensor(
        nums[None], device=DEVICE), rgb, lab


def _rollout_v2(net, seed, skill, n_bots, max_steps=250):
    import train_nn as T
    game = T._make_game(skill, seed, n_bots=n_bots)
    size = int(os.environ.get("V2_SIZE", "256"))
    grid = size // 8
    ep = dict(rgb=[], nums=[], kind=[], cell=[], pct=[], logp=[], reward=[])
    prev = None
    done = False
    while game.tick < game.max_ticks and not done and len(ep["reward"]) < max_steps:
        st = game.state_for(1)
        if not st.self_blob:
            break
        x, nums_t, rgb, lab = _bundle_x(game, prev, size)
        with torch.no_grad():
            click, kind, pct, value = net(x, nums_t)
            pk = torch.softmax(kind[0], 0)
            pc = torch.softmax(click[0], 0)
            kind_i = int(torch.multinomial(pk, 1))
            cell_i = int(torch.multinomial(pc, 1))
            lp = float(torch.log(pk[kind_i] + 1e-8) +
                       (torch.log(pc[cell_i] + 1e-8) if kind_i != 2 else 0))
        cy, cx = divmod(cell_i, grid)
        kind_s = {0: "expand", 1: "attack", 2: "bank"}[kind_i]
        from bot.planner import ClickAction
        act = ClickAction(kind_s, (cx + 0.5) / grid * game.w,
                          (cy + 0.5) / grid * game.w,
                          float(pct[0]) * 100 if kind_i == 1 else 0.0, reason="ppo")
        ep["rgb"].append(rgb)
        ep["nums"].append(game.numeric_ctx(1))
        ep["kind"].append(kind_i)
        ep["cell"].append(cell_i)
        ep["pct"].append(float(pct[0]))
        ep["logp"].append(lp)
        area_b = int((game.world == 1).sum())
        kills_b = game.players[1].kills
        actions = {1: game._clicks_for(act, 14)}
        for pid in game._pids:
            if pid == 1 or not game.players[pid].alive:
                continue
            actions[pid] = game._bot_clicks(pid)
        game.step(actions)
        area_a = int((game.world == 1).sum())
        rw = (area_a - area_b) / 2000.0
        if game.players[1].kills > kills_b:
            rw += 2.0
        if game.players[1].alive:
            rw += 0.005
            if area_a <= area_b:
                rw -= 0.002
        else:
            rw -= 1.0
            done = True
        ep["reward"].append(rw)
        prev = r = None
        prev_rgb = rgb
        prev = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    ep["alive"] = bool(game.players[1].alive)
    return ep, game


def _ep_tensors(ep):
    rgb = np.stack(ep["rgb"]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    prev = np.concatenate([rgb[:1], rgb[:-1]])
    diff = np.clip(rgb - prev, -1, 1) * 0.5 + 0.5
    x = torch.tensor(np.concatenate([rgb, diff], 1), device=DEVICE)
    nums = torch.tensor(np.stack(ep["nums"]), device=DEVICE)
    return (x, nums,
            torch.tensor(ep["kind"], device=DEVICE),
            torch.tensor(ep["cell"], device=DEVICE),
            torch.tensor(ep["pct"], device=DEVICE),
            torch.tensor(ep["logp"], device=DEVICE),
            np.array(ep["reward"], np.float32), ep["alive"])


def stage_ppo_v2(rounds=8, ep_per_round=6, epochs=4, gamma=0.99, lam=0.95,
                 clip=0.2, lr=3e-5):
    net = TeacherV2().to(DEVICE)
    if TEACH_PT.exists():
        net.load_state_dict(torch.load(TEACH_PT, map_location=DEVICE))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    base = net.module if hasattr(net, "module") else net
    opt = torch.optim.Adam(base.parameters(), lr=lr)
    best_wr = -1.0
    for rnd in range(rounds):
        eps = []
        for e in range(ep_per_round):
            ep, game = _rollout_v2(base, 1000 + rnd * 31 + e,
                                   np.random.choice(["easy", "medium", "hard"]),
                                   int(np.random.choice([6, 8, 10, 12])))
            if len(ep["reward"]) >= 10:
                eps.append(ep)
        if not eps:
            continue
        # advantages
        items = []
        alladv = []
        for ep in eps:
            x, nums, kb, cb, pb, oldlp, rw, alive = _ep_tensors(ep)
            with torch.no_grad():
                _, _, _, vals = net(x, nums)
                vals = vals.cpu().numpy()
            T_ = len(rw)
            rw10 = rw * 10.0
            adv = np.zeros(T_); gae = 0.0
            for t in reversed(range(T_)):
                nxt = vals[t + 1] if t + 1 < T_ else (1.0 if alive else 0.0)
                d = rw10[t] + gamma * nxt - vals[t]
                gae = d + gamma * lam * gae
                adv[t] = gae
            alladv += list(adv)
            items.append((x, nums, kb, cb, oldlp, adv))
        alladv = np.array(alladv)
        if alladv.std() > 1e-6:
            norm = (alladv - alladv.mean()) / (alladv.std() + 1e-8)
        else:
            norm = alladv
        off = 0
        for _ in range(epochs):
            for (x, nums, kb, cb, oldlp, adv) in items:
                T_ = len(adv)
                a_t = torch.tensor(norm[off:off + T_], device=DEVICE)
                off += T_
                click, kind, pct, value = net(x, nums)
                lpk = torch.log_softmax(kind, 1).gather(
                    1, kb.unsqueeze(1)).squeeze(1)
                lpc = torch.log_softmax(click, 1).gather(
                    1, cb.unsqueeze(1)).squeeze(1)
                lp = lpk + lpc * (kb != 2).float()
                ratio = torch.exp(lp - oldlp)
                pg = -torch.min(ratio * a_t,
                                torch.clamp(ratio, 1 - clip, 1 + clip) * a_t).mean()
                ent = -(torch.log_softmax(kind, 1) *
                        torch.log_softmax(kind, 1).exp()).mean()
                loss = pg - 0.03 * ent
                if not torch.isfinite(loss):
                    opt.zero_grad()
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
        wr = eval_v2(base, seeds=3)
        print(f"[ppo] round {rnd}: wr={wr:.2f}", flush=True)
        if wr > best_wr:
            best_wr = wr
            sd = base.state_dict()
            torch.save(sd, TEACH_PT)
            _upload_teacher()
    print(f"[ppo] done best_wr={best_wr:.2f}", flush=True)


def eval_v2(net, seeds=4):
    import train_nn as T
    wins = 0
    for s in range(seeds):
        ep, game = _rollout_v2(net, 5000 + s, "mixed", 10, max_steps=2400)
        alive = game.players[1].alive
        alive_n = sum(1 for p in game._pids if game.players[p].alive)
        wins += 1 if (alive and alive_n == 1) else 0
    return wins / seeds


def _upload_teacher():
    try:
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN", "")
        if not tok:
            return
        HfApi(token=tok).upload_file(
            path_or_fileobj=str(TEACH_PT), path_in_repo="v2/teacher.pt",
            repo_id="amer224/territorial-bot-data", repo_type="dataset", token=tok)
        print("[ppo] teacher uploaded", flush=True)
    except Exception as e:
        print("[ppo] upload fail", str(e)[:80], flush=True)
