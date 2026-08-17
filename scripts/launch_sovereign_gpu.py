#!/usr/bin/env python3
"""Launch SOVEREIGN v3 (290M, rival architecture + pipeline plumbing)
supervised Stage-A pretrain on Kaggle T4x2 — ONLY after the gate:
  HF rl/shards_v2 >= 15.0 GB  AND  /tmp/review_ok exists.

2026-08-14: user decision — SOVEREIGN replaces TeacherV3 as the GPU brain.
2026-08-15: boot now tees ALL stdout to /tmp/sovboot.log and uploads it to
  rl/sovereign_boot.log in try/finally — Kaggle stdout is invisible to us,
  and v1 of this kernel exited silently in 35 min with no heartbeat.

Usage: HF_TOKEN=... GH_TOKEN=... python3 scripts/launch_sovereign_gpu.py
GPU use = training only (user rule). One kernel, one purpose.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_loop_kernels import push_kernel  # noqa: E402

SOV_TRAIN_BOOT = '''import os, subprocess, sys, time, glob, shutil

LOGF = open("/tmp/sovboot.log", "w", buffering=1)

class _Tee:
    def __init__(self, f):
        self._f = f
        self._s = sys.__stdout__
    def write(self, t):
        try: self._f.write(t)
        except Exception: pass
        self._s.write(t)
    def flush(self):
        try: self._f.flush()
        except Exception: pass
        self._s.flush()

os.environ.setdefault("HF_TOKEN", "@@HF@@")
sys.stdout = _Tee(LOGF)
# v3: child subprocess output goes to the SAME log (v2 lost the traceback)
RUN = lambda *a, **k: subprocess.run(*a, stdout=LOGF,
                                     stderr=subprocess.STDOUT, **k)
print("sovereign-gpu boot v5", flush=True)

def _up_log():
    try:
        from huggingface_hub import upload_file
        upload_file(path_or_fileobj="/tmp/sovboot.log",
                    path_in_repo="rl/sovereign_boot.log",
                    repo_id="amer224/territorial-bot-data",
                    repo_type="dataset", token=os.environ["HF_TOKEN"])
        print("boot log uploaded", flush=True)
    except Exception as e:
        print("log upload fail:", str(e)[:120], flush=True)

try:
    RUN([sys.executable, "-m", "pip", "install", "-q",
         "torch==2.4.1",
         "--index-url", "https://download.pytorch.org/whl/cu121"],
        check=False)
    RUN([sys.executable, "-c",
         "import torch; a=torch.randn(16,16,device='cuda');"
         "(a@a).sum().item(); torch.cuda.synchronize();"
         "print('CUDA_OK gpus=', torch.cuda.device_count())"],
        check=False)
    RUN([sys.executable, "-m", "pip", "install", "-q", "numpy",
         "huggingface_hub", "pillow"], check=False)
    RUN(["git", "clone", "--depth", "1", "@@REPO_URL@@",
         "/kaggle/working/bot"], check=True)
    os.chdir("/kaggle/working/bot")
    # ---- data: ONE real-file download into the 20GB working disk --------
    # v1 died here: default cache lives on the small home disk and the
    # extra copy doubled the 17GB footprint. Symlinks off, no copy.
    os.environ["SOV_TMP"] = "/kaggle/working"
    SH = "/kaggle/working/shards/rl/shards_v2"
    os.makedirs(SH, exist_ok=True)
    # 2026-08-15: pool (20.5GB) EXCEEDS Kaggle's ~20GB disk -> run1/run2 died.
    # Download the NEWEST shards up to a 13GB cap instead of snapshot-all.
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=os.environ["HF_TOKEN"])
        fs = sorted((f for f in api.list_repo_tree(
            "amer224/territorial-bot-data", path_in_repo="rl/shards_v2",
            repo_type="dataset", token=os.environ["HF_TOKEN"])
            if getattr(f, "size", 0) > 0), key=lambda f: f.rfilename)
        cap = 13e9
        tot = 0
        n = 0
        for f in reversed(fs):
            if tot + f.size > cap:
                continue
            try:
                p = hf_hub_download("amer224/territorial-bot-data", f.rfilename,
                                    repo_type="dataset",
                                    local_dir="/kaggle/working/shards",
                                    local_dir_use_symlinks=False,
                                    token=os.environ["HF_TOKEN"])
                tot += f.size
                n += 1
            except Exception as e:
                print("dl skip", f.rfilename, str(e)[:60], flush=True)
        print(f"shards from HF: {n} ({tot/1e9:.1f} GB capped)", flush=True)
    except Exception as e:
        print("HF pull failed:", str(e)[:150], flush=True)
    # ---- GPU sanity smoke first (mini config, 3 steps, cheap) -----------
    r = RUN([sys.executable, "scripts/train_sovereign.py", "--mini",
             "--data-dir", SH,
             "--steps", "3", "--bs", "2", "--eval-seeds", "1",
             "--eval-every", "100000", "--max-minutes", "20",
             "--no-hf"])
    print("mini smoke rc=", r.returncode, flush=True)
    # ---- CUDA dry-run: 2 real steps so any GPU crash lands in the log ---
    r = RUN([sys.executable, "scripts/train_sovereign.py",
             "--device", "cuda", "--amp", "--bs", "8", "--grad-ckpt",
             "--data-dir", SH, "--steps", "2", "--eval-seeds", "1",
             "--eval-every", "999999", "--max-minutes", "10"])
    print("cuda dry-run rc=", r.returncode, flush=True)
    if r.returncode != 0:
        print("DRY-RUN FAILED — see log; NOT starting full train", flush=True)
        raise SystemExit(1)
    # ---- real Stage-A pretrain -------------------------------------------
    r = RUN([sys.executable, "scripts/train_sovereign.py",
             "--device", "cuda", "--amp", "--bs", "8",
             "--data-dir", SH,
                    # 2026-08-15: run1 stopped at step 758 with eval_every=800
                    # -> ZERO evals ever fired. 300 gives >=2 mid-run evals.
                    "--epochs", "@@EPOCHS@@", "--eval-every", "300",
             "--eval-seeds", "16", "--grad-ckpt"])
    print("sovereign stage-A rc=", r.returncode, flush=True)
    print("SOVEREIGN_GPU_DONE", flush=True)
finally:
    _up_log()
'''


def main():
    hf = os.environ.get("HF_TOKEN", "").strip()
    if not hf:
        print("HF_TOKEN required")
        sys.exit(1)
    # FULL MODE (17 Aug, user-approved 60h/wk): SOV_B=1 launches the acct2
    # aggressive-weights variant (survivor x2 / kill x4) as a parallel ablation.
    b = os.environ.get("SOV_B", "") == "1"
    code = (SOV_TRAIN_BOOT
            .replace("@@HF@@", hf)
            .replace("@@REPO_URL@@",
                     "https://" + os.environ.get("GH_TOKEN", "")
                     + "@github.com/amerameryou1-blip/bot.git")
            .replace("@@EPOCHS@@", os.environ.get("EPOCHS", "2")))
    code = ("import os\nos.environ['SOV_SURVIVOR_MULT']='2.0'\n"
            "os.environ['SOV_KILL_MULT']='4.0'\n" + code) if b else code
    slug = "sovereign-gpu-b" if b else "sovereign-gpu"
    owner = "amer38" if b else "amerameryou"
    tok = os.environ.get("KG2", "") if b else os.environ.get("KG1", "")
    print(push_kernel(slug, slug, code, gpu=True, owner=owner, token=tok))


if __name__ == "__main__":
    main()
