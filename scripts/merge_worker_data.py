#!/usr/bin/env python3
"""Pull all worker shards from HF and merge them into the local dataset.

Called by the main training notebook BEFORE vision/clone so the model trains
on all map types (island, mountains, desert, swamp) + the local lakes data.
"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np

HF_DATASET = "amer224/territorial-bot-data"
LOCAL = REPO / "weights" / "nn" / "dataset.npz"


def main():
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        print("HF_TOKEN required"); sys.exit(1)
    from huggingface_hub import hf_hub_download, HfApi
    api = HfApi(token=tok)
    try:
        files = api.list_repo_files(HF_DATASET, repo_type="dataset", token=tok)
    except Exception as e:
        print("no shards found:", e); return
    shards = sorted(f for f in files if f.startswith("shard_") and f.endswith(".npz"))
    print(f"found {len(shards)} shards on HF", flush=True)
    if not shards:
        return

    parts = []
    for s in shards:
        try:
            p = hf_hub_download(HF_DATASET, s, repo_type="dataset", token=tok)
            d = np.load(p)
            parts.append(d)
            print(f"  + {s}: {len(d['rgb'])} samples", flush=True)
        except Exception as e:
            print(f"  skip {s}: {e}", flush=True)
    if not parts:
        return
    keys = ["rgb", "seg", "centroid", "kind", "cell", "pct"]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    # merge with local if it exists
    if LOCAL.exists():
        loc = np.load(LOCAL)
        merged = {k: np.concatenate([merged[k], loc[k]]) for k in keys}
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LOCAL, **merged)
    print(f"merged dataset: {len(merged['rgb'])} total samples -> {LOCAL}", flush=True)


if __name__ == "__main__":
    main()
