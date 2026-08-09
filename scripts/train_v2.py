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
        new = [f for f in files if f.startswith("rl/shards_v2/")
               and not (R.SHARDS_V2 / Path(f).name).exists()
               and not (R.DONE_V2 / Path(f).name).exists()]
        for f in new[:4]:
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
            d = np.load(sp)
            rgb = d["rgb"]
            n = len(rgb)
            for i in range(0, n, bs):
                r = rgb[i:i + bs].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                prev = np.concatenate([r[:1], r[:-1]])
                diff = np.clip(r - prev, -1, 1) * 0.5 + 0.5
                x = np.concatenate([r, diff], 1)
                lab = d["lab"][i:i + bs]
                lab16 = torch.tensor(
                    _nn_resize(lab, GRID), dtype=torch.int64, device=DEVICE)
                yield (torch.tensor(x, device=DEVICE), lab16,
                       torch.tensor(d["nums"][i:i + bs], device=DEVICE),
                       torch.tensor(d["cell"][i:i + bs], device=DEVICE),
                       torch.tensor(d["kind"][i:i + bs], device=DEVICE),
                       torch.tensor(d["pct"][i:i + bs], device=DEVICE))


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
    torch.save(net.state_dict(), TEACH_PT)
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
    print("V2 DONE", flush=True)


if __name__ == "__main__":
    main()
