#!/usr/bin/env python3
"""FULL DATA AUDIT (user order 2026-08-09): verify recordings + shards
BEFORE training. Eyes + numbers. Prints PASS/FAIL per check.

Usage:
  python3 scripts/audit_data.py --recordings DIR            # audit local folder
  python3 scripts/audit_data.py --recordings DIR --purge    # also delete FAIL
                                                            # sessions from HF
  python3 scripts/audit_data.py --shards 5                  # audit N newest
                                                            # rl shards from HF
A click is UI-GARBAGE if inside: top banner (y<50), leaderboard (x<300,y<320),
bottom bar (y>740) — those opened modals / did nothing (match #6 post-mortem).
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

CROWN = (240, 224, 112)


def ui_zone(x, y):
    return y < 50 or (x < 300 and y < 320) or y > 740


def audit_session(sess: Path):
    fails, warns = [], []
    meta_p = sess / "meta.json"
    if not meta_p.exists():
        return ["no meta.json"], []
    meta = json.load(open(meta_p))
    sc = meta.get("self_color") or [0, 0, 0]
    if not meta.get("camera_pass"):
        fails.append("camera gate failed")
    if sum(abs(a - b) for a, b in zip(sc, CROWN)) < 90:
        fails.append("self_color is the CROWN icon (pre-fix bug)")
    if sum(abs(a - b) for a, b in zip(sc, (48, 180, 24))) < 30:
        fails.append("self_color untrusted (row-highlight green, pre-fix)")
    frames = sorted((sess / "frames").glob("*.jpg"))
    if not frames:
        fails.append("no frames")
        return fails, warns
    # self visibility over sampled frames
    covs = []
    lite = [int(v + (255 - v) * 0.55) for v in sc]
    for f in frames[:: max(1, len(frames) // 15)]:
        img = np.asarray(Image.open(f).convert("RGB")).astype(int)
        m = (np.abs(img - np.array(sc)).max(axis=2) < 24) | \
            (np.abs(img - np.array(lite)).max(axis=2) < 28)
        covs.append(float(m.mean()))
    med = float(np.median(covs))
    if med < 0.004:
        fails.append(f"bot invisible (median self coverage {med:.4f})")
    elif med < 0.01:
        warns.append(f"bot barely visible ({med:.4f})")
    # click validity
    bad = tot = 0
    for line in (sess / "clicks.jsonl").read_text().strip().split("\n"):
        if not line.strip():
            continue
        c = json.loads(line)
        tot += 1
        if ui_zone(c["x"], c["y"]) or not (0 <= c["x"] < 1280 and 0 <= c["y"] < 800):
            bad += 1
    if tot == 0:
        warns.append("no clicks")
    elif bad / tot > 0.2:
        fails.append(f"{bad}/{tot} clicks in UI zones (modal garbage)")
    elif bad:
        warns.append(f"{bad}/{tot} clicks in UI zones (filtered at label time)")
    return fails, warns


def audit_shard(path):
    d = np.load(path)
    probs = []
    rgb = d["rgb"]
    if rgb.dtype == np.uint8:
        pass  # uint8 by design — trainer/unpack divide by 255
    elif rgb.dtype == np.float32 and rgb.max() <= 1.0 + 1e-3 and rgb.min() >= 0:
        pass
    else:
        probs.append(f"rgb range bad ({rgb.min():.2f}..{rgb.max():.2f})")
    if int(d["lens"].sum()) != len(rgb):
        probs.append("lens sum != rgb len")
    if not np.isin(d["kind"], [0, 1, 2]).all():
        probs.append("kind out of range")
    if (d["cell"].min() < 0) or (d["cell"].max() > 255):
        probs.append("cell out of range")
    if (d["pct"].min() < 0) or (d["pct"].max() > 1):
        probs.append("pct out of range")
    if not (np.isfinite(d["reward"]).all() and np.isfinite(d["logp"]).all()):
        probs.append("non-finite reward/logp")
    return probs


def audit_v2_shard(path):
    d = np.load(path)
    probs = []
    if d["rgb"].dtype != np.uint8:
        probs.append("rgb not uint8")
    if d["rgb"].shape[1:] != (128, 128, 3):
        probs.append(f"rgb shape {d['rgb'].shape}")
    if not np.isin(d["lab"], [0, 1, 2, 3]).all():
        probs.append("lab out of range")
    if not np.isin(d["kind"], [0, 1, 2]).all():
        probs.append("kind out of range")
    if (d["cell"].min() < 0) or (d["cell"].max() > 255):
        probs.append("cell out of range")
    if int(d["lens"].sum()) != len(d["rgb"]):
        probs.append("lens sum != rgb len")
    if not (np.isfinite(d["reward"]).all() and np.isfinite(d["nums"]).all()):
        probs.append("non-finite reward/nums")
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings")
    ap.add_argument("--shards", type=int, default=0)
    ap.add_argument("--v2", type=int, default=0, help="audit N newest v2 shards")
    ap.add_argument("--purge", action="store_true")
    a = ap.parse_args()

    if a.v2:
        tok = os.environ.get("HF_TOKEN", "").strip()
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=tok)
        fs = sorted(f for f in api.list_repo_files(
            "amer224/territorial-bot-data", repo_type="dataset", token=tok)
            if f.startswith("rl/shards_v2/"))[-a.v2:]
        bad = 0
        for f in fs:
            p = hf_hub_download("amer224/territorial-bot-data", f,
                                repo_type="dataset", token=tok)
            probs = audit_v2_shard(p)
            bad += bool(probs)
            print(f"[{'FAIL' if probs else 'PASS'}] {f}: {'; '.join(probs) or 'clean'}")
        print("V2 AUDIT DONE", "— CLEAN" if not bad else f"— {bad} BAD")
        return

    PURGE_MARKS = ("camera gate failed", "CROWN", "invisible", "untrusted")
    bad_sessions, purge_list = [], []
    if a.recordings:
        for sess in sorted(Path(a.recordings).glob("*/meta.json")):
            s = sess.parent
            fails, warns = audit_session(s)
            tag = "FAIL" if fails else ("WARN" if warns else "PASS")
            print(f"[{tag}] {s.name}: {'; '.join(fails + warns) or 'clean'}")
            if fails:
                bad_sessions.append(s.name)
                if any(m in ";".join(fails) for m in PURGE_MARKS):
                    purge_list.append(s.name)
    if a.shards:
        tok = os.environ.get("HF_TOKEN", "").strip()
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=tok)
        fs = sorted(f for f in api.list_repo_files(
            "amer224/territorial-bot-data", repo_type="dataset", token=tok)
            if f.startswith("rl/shards/"))[-a.shards:]
        for f in fs:
            p = hf_hub_download("amer224/territorial-bot-data", f,
                                repo_type="dataset", token=tok)
            probs = audit_shard(p)
            print(f"[{'FAIL' if probs else 'PASS'}] {f}: {'; '.join(probs) or 'clean'}")
    # purge only unrecoverable sessions (invisible bot / wrong self color /
    # failed camera). Garbage-CLICK sessions stay: frames still train vision,
    # bad clicks are filtered at label time.
    if a.purge and purge_list:
        bad_sessions = purge_list
        tok = os.environ.get("HF_TOKEN", "").strip()
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        allf = api.list_repo_files("amer224/territorial-bot-data",
                                   repo_type="dataset", token=tok)
        for sid in bad_sessions:
            doomed = [f for f in allf if f.startswith(f"recordings/{sid}/")]
            if not doomed:
                continue
            try:
                api.delete_folder(path_in_repo=f"recordings/{sid}",
                                  repo_id="amer224/territorial-bot-data",
                                  repo_type="dataset", token=tok,
                                  commit_message=f"audit purge {sid}")
            except Exception:
                for p in doomed:
                    try:
                        api.delete_file(path_in_repo=p,
                                        repo_id="amer224/territorial-bot-data",
                                        repo_type="dataset", token=tok)
                    except Exception as e2:
                        print(f"  del fail {p}: {str(e2)[:60]}")
            print(f"purged {sid} ({len(doomed)} files)")
    print("AUDIT DONE")


if __name__ == "__main__":
    main()
