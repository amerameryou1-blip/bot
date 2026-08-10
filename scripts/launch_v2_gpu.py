#!/usr/bin/env python3
"""Launch the GPU teacher supervised-pretrain (ONLY after REVIEW.md gates).

Usage: HF_TOKEN=... KAGGLE_CONFIG_DIR=... python3 scripts/launch_v2_gpu.py
GPU use = training only (user rule). One kernel, one purpose.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launch_loop_kernels import push_kernel, GH_REPO_URL

V2_TRAIN_BOOT = '''import os, subprocess, sys, time
os.environ.setdefault("HF_TOKEN", "@@HF@@")
print("v2-gpu boot", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
                "--index-url", "https://download.pytorch.org/whl/cu121"],
               check=False)
subprocess.run([sys.executable, "-c",
                "import torch; a=torch.randn(16,16,device='cuda');"
                "(a@a).sum().item(); torch.cuda.synchronize(); print('CUDA_OK')"],
               check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy",
                "huggingface_hub"], check=False)
subprocess.run(["git", "clone", "--depth", "1", "@@REPO@@",
                "/kaggle/working/bot"], check=True)
os.chdir("/kaggle/working/bot")
# sprint data route: shards live in the private GH repo (HF was jammed)
subprocess.run(["git", "clone", "-q",
                "https://@@GHTOK@@@github.com/amerameryou1-blip/bot-recordings.git",
                "/tmp/shards_gh"], check=False)
import glob, shutil, os as _os
_os.makedirs("weights/nn/rl/shards_v2", exist_ok=True)
for f in glob.glob("/tmp/shards_gh/shard_v2_*.npz"):
    shutil.copy(f, "weights/nn/rl/shards_v2/")
print("shards from GH:", len(glob.glob("weights/nn/rl/shards_v2/*.npz")))
os.environ["EPOCHS"] = "@@EPOCHS@@"
r = subprocess.run([sys.executable, "scripts/train_v2.py", "sup"])
if r.returncode == 0:
    print("sup done; distill skipped (option B)")
# upload results back to HF
subprocess.run([sys.executable, "-c", """
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
for p in ['weights/nn/v2/teacher.pt', 'weights/nn/v2/student.pt']:
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo='v2/' + p.split('/')[-1],
                        repo_id='amer224/territorial-bot-data',
                        repo_type='dataset', token=os.environ['HF_TOKEN'])
        print('uploaded', p)
"""])
print("V2_GPU_DONE", flush=True)
'''


def main():
    hf = os.environ.get("HF_TOKEN", "").strip()
    if not hf:
        print("HF_TOKEN required")
        sys.exit(1)
    code = (V2_TRAIN_BOOT.replace("@@HF@@", hf)
            .replace("@@REPO@@", GH_REPO_URL)
            .replace("@@EPOCHS@@", os.environ.get("EPOCHS", "6"))
            .replace("@@GHTOK@@@", os.environ.get("GH_TOKEN", "")))
    print(push_kernel("v2-teacher-gpu", "v2-teacher-gpu", code, gpu=True))


if __name__ == "__main__":
    main()
