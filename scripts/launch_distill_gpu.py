#!/usr/bin/env python3
"""Launch distillation kernel (winner 290M -> nano). Env at push time:
DISTILL_CKPT (HF path), OUT_NAME, KG2=1 for acct2. GPU=training (allowed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_loop_kernels import push_kernel

BOOT = '''import os, subprocess, sys, glob, shutil
os.environ.setdefault("HF_TOKEN", "@@HF@@")
print("distill boot", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"],
               check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy",
                "huggingface_hub", "pillow"], check=False)
subprocess.run(["git", "clone", "--depth", "1", "@@REPO_URL@@",
                "/kaggle/working/bot"], check=True)
os.chdir("/kaggle/working/bot")
os.makedirs("weights/nn/rl/sov_shards", exist_ok=True)
try:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ["HF_TOKEN"])
    fs = sorted((f for f in api.list_repo_tree(
        "amer224/territorial-bot-data", path_in_repo="rl/shards_v2",
        repo_type="dataset", token=os.environ["HF_TOKEN"])
        if getattr(f, "size", 0) > 0), key=lambda f: f.rfilename)
    tot = 0; n = 0
    for f in reversed(fs):
        if tot + f.size > 6e9:
            continue
        hf_hub_download("amer224/territorial-bot-data", f.rfilename,
                        repo_type="dataset", local_dir="weights",
                        local_dir_use_symlinks=False,
                        token=os.environ["HF_TOKEN"])
        tot += f.size; n += 1
    print("distill shards:", n, round(tot/1e9,1), "GB", flush=True)
except Exception as e:
    print("shard pull fail:", str(e)[:120], flush=True)
env = dict(os.environ)
env["DISTILL_CKPT"] = "@@CKPT@@"
env["OUT_NAME"] = "@@OUT@@"
env["SH"] = "weights/rl/shards_v2"
r = subprocess.run([sys.executable, "scripts/distill_gpu.py"], env=env)
print("distill rc=", r.returncode, flush=True)
'''


def main():
    hf = os.environ.get("HF_TOKEN", "").strip()
    b = os.environ.get("KG2", "") == "1"
    code = (BOOT.replace("@@HF@@", hf)
            .replace("@@REPO_URL@@", "https://" + os.environ.get("GH_TOKEN", "")
                     + "@github.com/amerameryou1-blip/bot.git")
            .replace("@@CKPT@@", os.environ.get("DISTILL_CKPT",
                     "rl/sovereign/ckpt_final.pt"))
            .replace("@@OUT@@", os.environ.get("OUT_NAME", "nano_distill.pt")))
    slug = "sov-distill-b" if b else "sov-distill"
    owner = "amer38" if b else "amerameryou"
    tok = os.environ.get("KG2T", "") if b else os.environ.get("KG1T", "")
    print(push_kernel(slug, slug, code, gpu=True, owner=owner, token=tok))


if __name__ == "__main__":
    main()
