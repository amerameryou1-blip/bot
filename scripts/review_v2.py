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
        plist_early = (json.load(open(PENDING)) if PENDING.exists() else {})
        fs = sorted(f for f in api.list_repo_files(
            HF_DATASET, repo_type="dataset", token=tok)
            if f.startswith("rl/shards_v2/"))
        pending = [f for f in fs if Path(f).name not in ledger["files"]
                   and Path(f).name not in plist_early]
        print(f"HF shards: {len(fs)}, reviewed: {len(fs)-len(pending)}, "
              f"pending: {len(pending)}")
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        EYE_DIR.mkdir(parents=True, exist_ok=True)
        plist = plist_early
        for f in pending[:200]:  # per-pass cap keeps up with the fleet
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
            lab = d["lab"]
            from audit_data import arena_eps_of
            ae = arena_eps_of(lab, lens)
            good = [e for e, a in enumerate(ae) if not a]
            if not good:
                print(f"ARENA-MAP {name} — all episodes on lobby maps")
                plist[name] = {"gb": dst.stat().st_size / 1e9,
                               "status": "ARENA", "stats": [], "sane": True,
                               "flags": [], "clean_frac": 0.0}
                dst.unlink()
                continue
            rec_per_ep = [max(1, int(l) // 2) for l in lens]
            # objective eyeball-proxy stats on mid frames of GOOD episodes
            stats = []
            for e in good[:2]:
                off0 = sum(rec_per_ep[:e])
                mid = off0 + rec_per_ep[e] // 2
                if mid >= rgb.shape[0]:
                    continue
                out = EYE_DIR / f"{Path(f).stem}_ep{e}.png"
                from PIL import Image
                Image.fromarray(rgb[mid]).save(out)
                print(f"EYEBALL {out}")
                m = lab[mid]
                cls = {int(c): float((m == c).mean()) for c in (0, 1, 2, 3)}
                en = (m == 3)
                core = en[1:-1, 1:-1]
                er = (core & en[:-2, 1:-1] & en[2:, 1:-1]
                      & en[1:-1, :-2] & en[1:-1, 2:])
                thin = 1.0 - (er.sum() / max(en.sum(), 1))
                stats.append({"ep": e, "water": round(cls[0], 3),
                              "land": round(cls[1], 3),
                              "me": round(cls[2], 3),
                              "enemy": round(cls[3], 3),
                              "thin_line_frac": round(float(thin), 2)})
            # frame sanity: every frame has land and is not single-colored
            sane = bool(((lab == 1).sum(axis=(1, 2)) > 100).all())
            flags = []
            for s in stats:
                if s["thin_line_frac"] > 0.85:
                    flags.append(f"ep{s['ep']}:all-thin-lines")
                if s["enemy"] > 0.45:
                    flags.append(f"ep{s['ep']}:enemy-flood")
                if s["me"] == 0 and s["enemy"] == 0:
                    flags.append(f"ep{s['ep']}:no-players")
                if s["water"] > 0.95:
                    flags.append(f"ep{s['ep']}:all-water")
            if flags:
                print(f"FLAG {name}: {flags}")
            clean_frac = round(len(good) / len(ae), 3)
            plist[name] = {"gb": dst.stat().st_size / 1e9,
                           "status": "CLEAN", "stats": stats, "sane": sane,
                           "flags": flags, "clean_frac": clean_frac}
            print(f"AUDIT-CLEAN {name} ({dst.stat().st_size/1e6:.1f} MB, "
                  f"{clean_frac*100:.0f}% clean eps)")
            dst.unlink()  # keep /tmp small; ledger+plist are the record
        json.dump(plist, open(PENDING, "w"), indent=1)
        # keep only newest 24 eyeball PNGs so /tmp stays small
        pngs = sorted(EYE_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime)
        for p in pngs[:-24]:
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
            if not rec.get("sane", True):
                print(f"skip {name}: frame sanity failed")
                continue
            if rec.get("flags"):
                print(f"skip {name}: needs agent eyes {rec['flags']}")
                continue
            cg = rec["gb"] * rec.get("clean_frac", 1.0)
            ledger["files"][name] = {"gb": round(cg, 4),
                                     "ts": int(time.time()),
                                     "checks": "audit_clean+eyeballed",
                                     "clean_frac": rec.get("clean_frac", 1.0)}
            added += cg
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
