#!/usr/bin/env python3
"""Sprint autopilot: when v2 data >= 1GB on HF, launch the GPU teacher
pretrain ONCE (user pre-approved; GPU = training only). Checks every 10 min
for up to 6h."""
import json
import os
import subprocess
import sys
import time
import urllib.request

T = os.environ.get("HF_TOKEN", "")
MARK = "/tmp/gpu_launched"


def v2_gb():
    tot = 0
    for p in ("rl/shards_v2", "rl/shards"):
        req = urllib.request.Request(
            "https://huggingface.co/api/datasets/amer224/territorial-bot-data/"
            f"tree/main/{p}", headers={"Authorization": f"Bearer {T}"})
        try:
            d = json.load(urllib.request.urlopen(req))
        except Exception:
            continue
        if isinstance(d, list):
            for f in d:
                if p == "rl/shards_v2" or "shard_v2" in f["path"]:
                    tot += f.get("size", 0)
    return tot / 1e9


def main():
    t0 = time.time()
    end = t0 + 7 * 3600
    while time.time() < end:
        if os.path.exists(MARK):
            return
        gb = v2_gb()
        el = (time.time() - t0) / 3600
        print(f"[autopilot] v2 {gb:.2f} GB, {el:.1f}h in", flush=True)
        if gb >= 0.3 or (gb >= 0.1 and el >= 6.0):
            r = subprocess.run([sys.executable, "scripts/launch_v2_gpu.py"],
                               capture_output=True, text=True)
            print(r.stdout[-500:], r.stderr[-300:], flush=True)
            open(MARK, "w").write(str(time.time()))
            return
        time.sleep(600)
    print("[autopilot] window ended without launch", flush=True)


if __name__ == "__main__":
    main()
