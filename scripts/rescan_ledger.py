#!/usr/bin/env python3
"""2026-08-14: recompute TRUE clean fraction of every ledger shard with the
fixed arena detector (lobby-UI template flags). Streams shards from HF,
computes clean_frac, rewrites ledger weights + total_gb honestly.

Usage: HF_TOKEN=... python3 scripts/rescan_ledger.py [--limit N]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_data import arena_eps_of  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "weights/nn/rl/reviewed_ledger.json"
HF = "amer224/territorial-bot-data"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tok = os.environ.get("HF_TOKEN", "")
    from huggingface_hub import hf_hub_download
    ledger = json.load(open(LEDGER))
    names = list(ledger["files"])
    if limit:
        names = names[:limit]
    new_total = 0.0
    done = 0
    for name in names:
        rec = ledger["files"][name]
        raw = rec["gb"] / max(rec.get("clean_frac", 1.0), 1e-6)
        try:
            p = hf_hub_download(HF, f"rl/shards_v2/{name}",
                                repo_type="dataset", token=tok,
                                local_dir="/tmp/rescan")
            d = np.load(p, allow_pickle=True)
            ae = arena_eps_of(d["lab"], d["lens"])
            frac = round(float(1.0 - np.mean(ae)), 3) if len(ae) else 1.0
        except Exception as e:
            print(f"skip {name}: {str(e)[:60]}", flush=True)
            frac = rec.get("clean_frac", 1.0)
        rec["clean_frac"] = frac
        rec["gb"] = round(raw * frac, 4)
        new_total += rec["gb"]
        done += 1
        if done % 25 == 0:
            print(f"rescan {done}/{len(names)} running total {new_total:.3f} GB",
                  flush=True)
    # untouched entries (beyond limit) keep their weight
    for name in ledger["files"]:
        if name not in names:
            new_total += ledger["files"][name]["gb"]
    ledger["total_gb"] = round(new_total, 4)
    json.dump(ledger, open(LEDGER, "w"), indent=1)
    open("/tmp/reviewed_gb", "w").write(str(ledger["total_gb"]))
    print(f"RESCAN DONE: true reviewed total = {ledger['total_gb']:.3f} GB",
          flush=True)


if __name__ == "__main__":
    main()
