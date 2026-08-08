#!/usr/bin/env python3
"""Launch the continuous sim-RL loop on Kaggle (HANDOFF §14 v1).

  5x CPU workers  (rl-worker-1..5) — play sims with latest ckpt, push shards
  1x GPU trainer  (rl-trainer)     — PPO on new shards, push checkpoints
                                     (delayed start so workers fill HF first)

Tokens are injected into TEMP kernel copies at push time (private kernels);
the GitHub repo stays token-free. GPU is used ONLY for training.

Usage:
  KAGGLE_CONFIG_DIR=... python3 scripts/launch_loop_kernels.py [--workers 5]
  HF_TOKEN=... (injected into kernels)
"""
import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GH_REPO_URL = "https://github.com/amerameryou1-blip/bot.git"

WORKER_BOOT = '''import os, subprocess, time
os.environ.setdefault("HF_TOKEN", "@@HF@@")
print("worker boot", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch",
                "--index-url", "https://download.pytorch.org/whl/cpu"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy",
                "huggingface_hub"], check=False)
subprocess.run(["git", "clone", "--depth", "1", "@@REPO@@", "/kaggle/working/bot"],
               check=True)
os.chdir("/kaggle/working/bot")
os.environ["EP_PER_SHARD"] = "10"
os.environ["SHARD_SLEEP_S"] = "10"
subprocess.run([sys.executable, "scripts/rl_loop.py", "worker",
                "--hours", "@@HOURS@@", "--skill", "medium", "--n-bots", "8"],
               check=False)
print("WORKER_SESSION_DONE", flush=True)
'''

TRAINER_BOOT = '''import os, sys, subprocess, time
os.environ.setdefault("HF_TOKEN", "@@HF@@")
print("trainer boot (GPU, delayed @@DELAY@@min so workers fill HF first)", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy",
                "huggingface_hub", "safetensors"], check=False)
subprocess.run(["git", "clone", "--depth", "1", "@@REPO@@", "/kaggle/working/bot"],
               check=True)
os.chdir("/kaggle/working/bot")
time.sleep(@@DELAY@@ * 60)
subprocess.run([sys.executable, "scripts/rl_loop.py", "trainer",
                "--hours", "@@HOURS@@"], check=False)
print("TRAINER_SESSION_DONE", flush=True)
'''


def push_kernel(slug: str, title: str, code: str) -> str:
    d = tempfile.mkdtemp(prefix=f"kg-{slug}-")
    with open(os.path.join(d, "main.py"), "w") as f:
        f.write("import sys\n" + code)
    meta = {
        "id": f"amerameryou/{slug}",
        "title": title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": slug == "rl-trainer",
        "enable_internet": True,
        "kernel_sources": [],
        "dataset_sources": [],
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    r = subprocess.run(["kaggle", "kernels", "push", "-p", d],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().replace("\n", " ")
    shutil.rmtree(d, ignore_errors=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--worker-hours", type=float, default=8.5)
    ap.add_argument("--trainer-hours", type=float, default=9.0)
    ap.add_argument("--trainer-delay-min", type=int, default=40)
    ap.add_argument("--trainer-only", action="store_true")
    a = ap.parse_args()

    hf = os.environ.get("HF_TOKEN", "").strip()
    if not hf:
        print("HF_TOKEN required (injected into private kernels)"); sys.exit(1)

    if not a.trainer_only:
        for i in range(1, a.workers + 1):
            code = (WORKER_BOOT.replace("@@HF@@", hf).replace("@@REPO@@", GH_REPO_URL)
                    .replace("@@HOURS@@", str(a.worker_hours)))
            # title == slug so re-pushes update the SAME kernel
            out = push_kernel(f"rl-loop-worker-{i}", f"rl-loop-worker-{i}", code)
            print(f"pushed rl-loop-worker-{i}: {out[:160]}", flush=True)
    code = (TRAINER_BOOT.replace("@@HF@@", hf).replace("@@REPO@@", GH_REPO_URL)
            .replace("@@HOURS@@", str(a.trainer_hours))
            .replace("@@DELAY@@", str(a.trainer_delay_min)))
    out = push_kernel("rl-loop-trainer-gpu", "rl-loop-trainer-gpu", code)
    print(f"pushed rl-loop-trainer-gpu: {out[:160]}", flush=True)


if __name__ == "__main__":
    main()
