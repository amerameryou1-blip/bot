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
REVIEW = "/tmp/review_ok"


def gh_count():
    import subprocess
    r = subprocess.run(["git", "ls-remote", os.environ.get("GH_URL", "")],
                       capture_output=True, text=True)
    return 0  # unused; size checked via API below


def v2_count_gh():
    req = urllib.request.Request(
        "https://api.github.com/repos/amerameryou1-blip/bot-recordings/contents/",
        headers={"Authorization": "token " + os.environ.get("GH_TOKEN", "")})
    try:
        d = json.load(urllib.request.urlopen(req))
        return len([f for f in d if f["name"].startswith("shard_v2")])
    except Exception:
        return 0


def v2_gb():
    # 2026-08-15 fix: raw tree API is NOT paginated -> first page only
    # (~1000 files = 7.7GB) and the gate could never see >=15GB. Use the
    # paginated hub client.
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=T or None)
        tot = 0
        for p in ("rl/shards_v2", "rl/shards"):
            try:
                for f in api.list_repo_tree(
                        "amer224/territorial-bot-data", path_in_repo=p,
                        repo_type="dataset", token=T or None):
                    if getattr(f, "size", 0):
                        tot += f.size
            except Exception:
                pass
        return tot / 1e9
    except Exception:
        pass
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
    end = t0 + 36 * 3600
    while time.time() < end:
        if os.path.exists(MARK):
            return
        gb = v2_gb()
        el = (time.time() - t0) / 3600
        print(f"[autopilot] v2 {gb:.2f} GB, {el:.1f}h in", flush=True)
        n_gh = v2_count_gh()
        print(f"[autopilot] GH shards {n_gh}", flush=True)
        # user rule (2026-08-10): NO GPU unless >=15GB AND agent reviewed data
        # 2026-08-14: SOVEREIGN (290M rival brain + pipeline plumbing) is the
        # gate brain now; env GATE_BRAIN=v3 falls back to TeacherV3.
        if gb >= 15.0 and os.path.exists(REVIEW):
            launcher = ("scripts/launch_v2_gpu.py"
                        if os.environ.get("GATE_BRAIN") == "v3"
                        else "scripts/launch_sovereign_gpu.py")
            r = subprocess.run([sys.executable, launcher],
                               capture_output=True, text=True)
            print(r.stdout[-500:], r.stderr[-300:], flush=True)
            open(MARK, "w").write(str(time.time()))
            return
        time.sleep(600)
    print("[autopilot] window ended without launch", flush=True)


if __name__ == "__main__":
    main()
