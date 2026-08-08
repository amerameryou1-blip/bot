"""v15 full pipeline as a plain SCRIPT kernel (CPU-only).

The notebook route kept dying on Kaggle CPU VMs (preinstalled CUDA torch,
user-site installs invisible to stage subprocesses). A single script gives
full control: uninstall CUDA torch, install CPU torch system-wide, then run
the whole training pipeline. GPU hours stay untouched.
"""
import os
import shutil
import subprocess
import sys

HF = "@@HF@@"
REPO = "https://github.com/amerameryou1-blip/bot.git"

print("v15-cpu boot", flush=True)
if not shutil.which("nvidia-smi"):
    print("no GPU -> CPU torch (system-wide)", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch"],
                   check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cpu"],
                   check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy",
                "huggingface_hub", "safetensors", "pillow"], check=False)
subprocess.run(["git", "clone", "--depth", "1", REPO, "/kaggle/working/bot"],
               check=True)
os.chdir("/kaggle/working/bot")
os.environ["HF_TOKEN"] = HF
os.environ["FORCE_CPU"] = "1"
os.environ["PPO_ROUNDS"] = "20"
os.environ["WORKERS"] = "2"

env = dict(os.environ)
run = lambda *a: subprocess.run([sys.executable, *a], env=env)

# 1) merge worker shards from HF into dataset.npz
run("scripts/merge_worker_data.py")

# 2) pull real recordings + screenshots, label them (curated skip list)
pull = (
    "import os\n"
    "from huggingface_hub import snapshot_download\n"
    "p = snapshot_download('amer224/territorial-bot-data', repo_type='dataset',\n"
    "    allow_patterns=['recordings/*', 'screenshots/*'], token=os.environ['HF_TOKEN'])\n"
    "print('pulled', p)\n"
    "import glob, shutil\n"
    "os.makedirs('recordings', exist_ok=True)\n"
    "for m in glob.glob(os.path.join(p, 'recordings', '*', 'meta.json')):\n"
    "    s = os.path.dirname(m)\n"
    "    shutil.copytree(s, os.path.join('recordings', os.path.basename(s)), dirs_exist_ok=True)\n"
    "if os.path.exists(os.path.join(p, 'screenshots')):\n"
    "    shutil.copytree(os.path.join(p, 'screenshots'), 'realdata/shots', dirs_exist_ok=True)\n"
)
subprocess.run([sys.executable, "-c", pull], env=env)
run("scripts/label_real.py", "--recordings", "recordings",
    "--out", "weights/nn/real_vision.npz", "--save-anyway",
    "--skip", "20260808-164951,20260808-170008,20260808-170658")

# 3) the pipeline: seed(distill) -> vision -> clone -> real -> ppo -> eval
for stage in (["seed"], ["vision"], ["clone"], ["real"], ["ppo", "20"], ["eval"]):
    r = run("scripts/train_nn.py", *stage)
    print(f"stage {stage} rc={r.returncode}", flush=True)
    if r.returncode != 0 and stage[0] in ("seed", "vision"):
        print("FATAL early stage failed", flush=True)
        sys.exit(1)

# 4) export best model to HF model repo (recreate it)
run("scripts/export_hf.py")
print("V15_CPU_DONE", flush=True)
