#!/usr/bin/env python3
"""Data-worker: collect a SLICE of training data (map type + skill + seeds)
and upload it to the shared HF dataset repo `amer224/territorial-bot-data`.

Env: HF_TOKEN, MAP_TYPE, SKILL, SEED_START, SEED_END, WORKERS
"""
import os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np

HF_DATASET = "amer224/territorial-bot-data"


def main():
    map_type = os.environ.get("MAP_TYPE", "lakes")
    skill = os.environ.get("SKILL", "medium")
    seed_start = int(os.environ.get("SEED_START", "1"))
    seed_end = int(os.environ.get("SEED_END", "30"))
    workers = int(os.environ.get("WORKERS", "4"))
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN required"); sys.exit(1)

    from nn import data as ndata
    t0 = time.time()
    print(f"worker: map={map_type} skill={skill} seeds={seed_start}..{seed_end} workers={workers}", flush=True)
    data = ndata.collect_parallel(seeds=seed_end - seed_start + 1, n_bots=3, h=110, w=140,
                                  max_ticks=1400, clicks_per_tick=12, eps=0.15,
                                  bot_skill=skill, workers=workers, record_every=4,
                                  map_type=map_type)
    if data is None:
        print("no data collected"); sys.exit(1)
    n = len(data["rgb"])
    print(f"collected {n} samples in {time.time()-t0:.0f}s", flush=True)

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        api.create_repo(HF_DATASET, repo_type="dataset", private=True, exist_ok=True, token=token)
    except Exception as e:
        print("create_repo:", e, flush=True)
    fname = f"shard_{map_type}_{skill}_s{seed_start}-{seed_end}.npz"
    local = f"/tmp/{fname}"
    np.savez_compressed(local, **data)
    api.upload_file(path_or_fileobj=local, path_in_repo=fname, repo_id=HF_DATASET,
                    repo_type="dataset", token=token)
    print(f"uploaded {fname} ({os.path.getsize(local)/1e6:.1f} MB) -> {HF_DATASET}", flush=True)
    print("WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
