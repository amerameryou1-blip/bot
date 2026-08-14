#!/usr/bin/env python3
"""Launch SOVEREIGN v3 (290M, rival architecture + pipeline plumbing)
supervised Stage-A pretrain on Kaggle T4x2 — ONLY after the gate:
  HF rl/shards_v2 >= 15.0 GB  AND  /tmp/review_ok exists.

2026-08-14: user decision — SOVEREIGN replaces TeacherV3 as the GPU brain.
launch_v2_gpu.py stays as fallback (env GATE_BRAIN=v3 in auto_gpu_launch).

Usage: HF_TOKEN=... GH_TOKEN=... python3 scripts/launch_sovereign_gpu.py
GPU use = training only (user rule). One kernel, one purpose.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_loop_kernels import push_kernel  # noqa: E402

SOV_TRAIN_BOOT = '''import os, subprocess, sys, time, glob, shutil
os.environ.setdefault("HF_TOKEN", "@@HF@@")
print("sovereign-gpu boot", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"],
               check=False)
subprocess.run([sys.executable, "-c",
                "import torch; a=torch.randn(16,16,device='cuda');"
                "(a@a).sum().item(); torch.cuda.synchronize();"
                "print('CUDA_OK gpus=', torch.cuda.device_count())"],
               check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy",
                "huggingface_hub", "pillow"], check=False)
subprocess.run(["git", "clone", "--depth", "1", "@@REPO_URL@@",
                "/kaggle/working/bot"], check=True)
os.chdir("/kaggle/working/bot")
# ---- data: pull the FULL reviewed pool from HF --------------------------
os.makedirs("weights/nn/rl/sov_shards", exist_ok=True)
try:
    from huggingface_hub import snapshot_download
    p2 = snapshot_download("amer224/territorial-bot-data",
                           repo_type="dataset",
                           allow_patterns=["rl/shards_v2/*"],
                           token=os.environ["HF_TOKEN"])
    for f in glob.glob(p2 + "/rl/shards_v2/*.npz"):
        shutil.copy(f, "weights/nn/rl/sov_shards/")
    print("shards from HF:", len(glob.glob("weights/nn/rl/sov_shards/*.npz")))
except Exception as e:
    print("HF pull failed:", str(e)[:150])
# ---- GPU sanity smoke first (mini config, 3 steps, cheap) ---------------
r = subprocess.run([sys.executable, "scripts/train_sovereign.py", "--mini",
                    "--data-dir", "weights/nn/rl/sov_shards",
                    "--steps", "3", "--bs", "2", "--eval-seeds", "1",
                    "--eval-every", "100000", "--max-minutes", "20",
                    "--no-hf"])
print("mini smoke rc=", r.returncode, flush=True)
# ---- real Stage-A pretrain ----------------------------------------------
r = subprocess.run([sys.executable, "scripts/train_sovereign.py",
                    "--device", "cuda", "--amp", "--bs", "16",
                    "--data-dir", "weights/nn/rl/sov_shards",
                    "--epochs", "@@EPOCHS@@", "--eval-every", "800",
                    "--eval-seeds", "16", "--grad-ckpt"])
print("sovereign stage-A rc=", r.returncode, flush=True)
print("SOVEREIGN_GPU_DONE", flush=True)
'''


def main():
    hf = os.environ.get("HF_TOKEN", "").strip()
    if not hf:
        print("HF_TOKEN required")
        sys.exit(1)
    code = (SOV_TRAIN_BOOT
            .replace("@@HF@@", hf)
            .replace("@@REPO_URL@@",
                     "https://" + os.environ.get("GH_TOKEN", "")
                     + "@github.com/amerameryou1-blip/bot.git")
            .replace("@@EPOCHS@@", os.environ.get("EPOCHS", "2")))
    print(push_kernel("sovereign-gpu", "sovereign-gpu", code, gpu=True))


if __name__ == "__main__":
    main()
