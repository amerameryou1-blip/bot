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
KERNELS = [f"rl-loop-worker-{i}" for i in range(1, 6)] + ["rl-loop-trainer-gpu"]
ALIVE = ("RUNNING", "QUEUED")


def status(slug: str) -> str:
    r = subprocess.run(["kaggle", "kernels", "status", f"amerameryou/{slug}"],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr)
    for tok in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "RUNNING", "QUEUED"):
        if tok in out:
            return tok
    return out.strip()[:60]


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
            if k == "rl-loop-trainer-gpu":
                code = (L.TRAINER_BOOT.replace("@@HF@@", hf)
                        .replace("@@REPO@@", L.GH_REPO_URL)
                        .replace("@@HOURS@@", "9.0").replace("@@DELAY@@", "0"))
            else:
                code = (L.WORKER_BOOT.replace("@@HF@@", hf)
                        .replace("@@REPO@@", L.GH_REPO_URL)
                        .replace("@@HOURS@@", "8.5"))
            out = L.push_kernel(k, k, code)
            print(f"relaunched {k}: {out[:140]}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
