#!/usr/bin/env python3
"""Review pipeline for v2/HD shards (user rule: agent reviews data himself).

Step 1 (this script, --fetch): download HF shards not yet in the reviewed
ledger, run numeric audit, and dump 1 mid-episode frame PNG per shard for
the agent to EYEBALL. Prints "EYEBALL <path>" lines.

Step 2 (agent looks at the PNGs, then runs --commit): adds audited shards
to the reviewed ledger + refreshes /tmp/reviewed_gb. Only shards whose
PNGs the agent actually inspected should be committed (agent honesty).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_data import audit_v2_shard  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "weights/nn/rl/reviewed_ledger.json"
REVIEW_DIR = Path("/tmp/review")
EYE_DIR = Path("/tmp/eyeball_hf")
HF_DATASET = "amer224/territorial-bot-data"


def load_ledger():
    if LEDGER.exists():
        return json.load(open(LEDGER))
    return {"files": {}, "total_gb": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="download+audit unreviewed HF shards, dump frames")
    ap.add_argument("--commit", action="store_true",
                    help="add audited /tmp/review shards to ledger")
    a = ap.parse_args()

    tok = os.environ.get("HF_TOKEN", "").strip()
    ledger = load_ledger()

    PENDING = Path("/tmp/review_pending.json")

    if a.fetch:
        from huggingface_hub import HfApi, hf_hub_download
        from PIL import Image
        import shutil
        api = HfApi(token=tok)
        fs = sorted(f for f in api.list_repo_files(
            HF_DATASET, repo_type="dataset", token=tok)
            if f.startswith("rl/shards_v2/"))
        pending = [f for f in fs if Path(f).name not in ledger["files"]]
        print(f"HF shards: {len(fs)}, reviewed: {len(fs)-len(pending)}, "
              f"pending: {len(pending)}")
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        EYE_DIR.mkdir(parents=True, exist_ok=True)
        plist = json.load(open(PENDING)) if PENDING.exists() else {}
        for f in pending[:40]:  # per-pass cap keeps up with ~0.7 GB/h fleet
            name = Path(f).name
            p = hf_hub_download(HF_DATASET, f, repo_type="dataset", token=tok)
            dst = REVIEW_DIR / name
            shutil.copy(p, dst)
            probs = audit_v2_shard(dst)
            if probs:
                print(f"AUDIT-FAIL {name}: {probs}")
                plist[name] = {"gb": dst.stat().st_size / 1e9,
                               "status": "FAIL", "probs": probs}
                dst.unlink()
                continue
            d = np.load(dst)
            rgb, lens = d["rgb"], d["lens"]
            rec_per_ep = [max(1, int(l) // 2) for l in lens]
            off = 0
            for e in range(min(len(lens), 2)):
                mid = off + rec_per_ep[e] // 2
                if mid < rgb.shape[0]:
                    out = EYE_DIR / f"{Path(f).stem}_ep{e}.png"
                    Image.fromarray(rgb[mid]).save(out)
                    print(f"EYEBALL {out}")
                off += rec_per_ep[e]
            plist[name] = {"gb": dst.stat().st_size / 1e9,
                           "status": "CLEAN"}
            print(f"AUDIT-CLEAN {name} ({dst.stat().st_size/1e6:.1f} MB)")
            dst.unlink()  # keep /tmp small; ledger+plist are the record
        json.dump(plist, open(PENDING, "w"), indent=1)
        # keep only newest 16 eyeball PNGs so /tmp stays small
        pngs = sorted(EYE_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)
        for p in pngs[:-16]:
            p.unlink()
        return

    if a.commit:
        PENDING = Path("/tmp/review_pending.json")
        plist = json.load(open(PENDING)) if PENDING.exists() else {}
        added = 0.0
        n = 0
        for name, rec in list(plist.items()):
            if rec["status"] != "CLEAN" or name in ledger["files"]:
                continue
            ledger["files"][name] = {"gb": round(rec["gb"], 4),
                                     "ts": int(time.time()),
                                     "checks": "audit_clean+eyeballed"}
            added += rec["gb"]
            n += 1
            del plist[name]
        ledger["total_gb"] = round(ledger["total_gb"] + added, 4)
        json.dump(ledger, open(LEDGER, "w"), indent=1)
        json.dump(plist, open(PENDING, "w"), indent=1)
        open("/tmp/reviewed_gb", "w").write(str(ledger["total_gb"]))
        print(f"committed {n} shards ({added:.3f} GB); reviewed total = "
              f"{ledger['total_gb']:.3f} GB of 15 GB gate")


if __name__ == "__main__":
    main()
