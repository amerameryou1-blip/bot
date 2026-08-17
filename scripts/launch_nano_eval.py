#!/usr/bin/env python3
"""Launch CPU kernel that evaluates a distilled nano ckpt with the HONEST
16-seed last-survivor eval (nano_rollout.py). Env: NANO_CKPT (HF path,
e.g. rl/nano_v10_distill.pt), KG2=1 for acct2."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_loop_kernels import push_kernel

BOOT = '''import os, subprocess, sys
os.environ.setdefault("HF_TOKEN", "@@HF@@")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "--index-url",
                "https://download.pytorch.org/whl/cpu"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy",
                "huggingface_hub", "pillow"], check=False)
subprocess.run(["git", "clone", "--depth", "1", "@@REPO_URL@@",
                "/kaggle/working/bot"], check=True)
os.chdir("/kaggle/working/bot")
from huggingface_hub import hf_hub_download
p = hf_hub_download("amer224/territorial-bot-data", "@@CKPT@@",
                    repo_type="dataset", token=os.environ["@@HFENV@@"],
                    local_dir="/tmp/nk")
r = subprocess.run([sys.executable, "scripts/nano_rollout.py", "--ckpt", p,
                    "--seeds", "16"])
print("nano eval rc=", r.returncode, flush=True)
'''


def main():
    b = os.environ.get("KG2", "") == "1"
    hf = os.environ.get("HF_TOKEN", "").strip()
    code = (BOOT.replace("@@HF@@", hf)
            .replace("@@REPO_URL@@", "https://" + os.environ.get("GH_TOKEN", "")
                         + "@github.com/amerameryou1-blip/bot.git")
            .replace("@@CKPT@@", os.environ.get("NANO_CKPT",
                         "rl/nano_distill.pt"))
            .replace("@@HFENV@@", "HF_TOKEN"))
    slug = "nano-eval-b" if b else "nano-eval"
    owner = "amer38" if b else "amerameryou"
    tok = os.environ.get("KG2T", "") if b else os.environ.get("KG1T", "")
    print(push_kernel(slug, slug, code, gpu=False, owner=owner, token=tok))


if __name__ == "__main__":
    main()
