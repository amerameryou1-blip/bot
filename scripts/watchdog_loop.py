#!/usr/bin/env python3
"""Watchdog for the continuous RL loop on Kaggle.

Prints status of rl-worker-1..5 + rl-trainer; re-pushes any kernel that is
not RUNNING/QUEUED (Kaggle sessions cap at ~9-12h — the loop must resume).

Usage: KAGGLE_CONFIG_DIR=... python3 scripts/watchdog_loop.py [--relaunch]
"""
import os
import sys
import subprocess
import argparse

# Kaggle slugifies kernel titles: "RL loop worker N" -> rl-loop-worker-N
# fleet = 3 workers + CPU trainer + CPU v15 = exactly the 5-CPU cap
# (Kaggle enforces it server-side; stay AT it, never over — user TOS check
# 2026-08-08: CPU batch cap 5, GPU cap 2, separate pools)
KERNELS = ([f"rl-loop-worker-{i}" for i in range(1, 4)]
           + [f"rl-v2-worker-{i}" for i in (1, 2)]
           + ["rl-loop-trainer-cpu"])
ALIVE = ("RUNNING", "QUEUED")


def status(slug: str) -> str:
    r = subprocess.run(["kaggle", "kernels", "status", f"amerameryou/{slug}"],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    for tok in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "RUNNING", "QUEUED"):
        if tok in out:
            return tok
    return out.strip()[:60]


def push_trainer_best_effort():
    """Keep trying to push the newest trainer; rejected while 5 CPU sessions
    run, goes through the instant a slot frees (v15 finish / worker end)."""
    try:
        import launch_loop_kernels as L
        hf = os.environ.get("HF_TOKEN", "").strip()
        if not hf:
            return
        code = (L.TRAINER_BOOT.replace("@@HF@@", hf).replace("@@REPO@@", L.GH_REPO_URL)
                .replace("@@HOURS@@", "9.0").replace("@@DELAY@@", "1"))
        out = L.push_kernel("rl-loop-trainer-cpu", "rl-loop-trainer-cpu", code)
        print(f"trainer push: {out[:100]}", flush=True)
    except Exception as e:
        print(f"trainer push err: {e}", flush=True)


def tidy_old_shards():
    """Delete the 6 old root filler shards once HF's 128-commits/h budget
    allows (idempotent; silently retries every watchdog cycle)."""
    try:
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN", "").strip()
        if not tok:
            return
        api = HfApi(token=tok)
        fs = api.list_repo_files("amer224/territorial-bot-data", repo_type="dataset")
        old = [f for f in fs if f.startswith("shard_") and f.endswith(".npz")]
        for f in old:
            api.delete_file(path_in_repo=f, repo_id="amer224/territorial-bot-data",
                            repo_type="dataset", token=tok)
        if old:
            print(f"tidied {len(old)} old used shards", flush=True)
    except Exception as e:
        print(f"tidy skipped (budget?): {str(e)[:60]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relaunch", action="store_true",
                    help="re-push kernels that finished/died")
    a = ap.parse_args()
    dead = []
    for k in KERNELS:
        s = status(k)
        print(f"{k:14s} {s}", flush=True)
        if not any(x in s for x in ALIVE):
            dead.append(k)
    push_trainer_best_effort()
    tidy_old_shards()
    if a.relaunch and dead:
        # a CPU slot just freed -> ferry any recordings zips waiting on GH
        try:
            subprocess.run([sys.executable,
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "push_migrate.py")], timeout=2400)
        except Exception as e:
            print(f"migrate hop failed: {e}")
        import launch_loop_kernels as L
        hf = os.environ.get("HF_TOKEN", "").strip()
        for k in dead:
            if k == "rl-loop-trainer-cpu":
                code = (L.TRAINER_BOOT.replace("@@HF@@", hf)
                        .replace("@@REPO@@", L.GH_REPO_URL)
                        .replace("@@HOURS@@", "9.0").replace("@@DELAY@@", "0"))
            elif k.startswith("rl-v2-worker"):
                code = (L.V2_WORKER_BOOT.replace("@@HF@@", hf)
                        .replace("@@REPO@@", L.GH_REPO_URL)
                        .replace("@@HOURS@@", "8.5"))
            else:
                code = (L.WORKER_BOOT.replace("@@HF@@", hf)
                        .replace("@@REPO@@", L.GH_REPO_URL)
                        .replace("@@HOURS@@", "8.5"))
            out = L.push_kernel(k, k, code)
            print(f"relaunched {k}: {out[:140]}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
