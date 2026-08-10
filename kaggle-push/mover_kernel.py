"""MOVER: rescue v2 shards from finished workers' outputs -> HF.
Runs on Kaggle (fast net), 1 commit per worker dir. Injected: @@HF@@ @@KG@@"""
import json
import os
import subprocess
import sys

os.environ["HF_TOKEN"] = "@@HF@@"
os.makedirs("/root/.kaggle", exist_ok=True)
json.dump({"username": "amerameryou", "key": "@@KG@@"},
          open("/root/.kaggle/kaggle.json", "w"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle",
                "huggingface_hub"], check=False)

for w in (1, 2, 3, 4):
    slug = f"amerameryou/rl-v2-worker-{w}"
    out = f"/tmp/out{w}"
    r = subprocess.run(["kaggle", "kernels", "output", slug, "-p", out, "--force"],
                       capture_output=True, text=True)
    print(slug, "->", (r.stdout + r.stderr).strip()[-120:])
    d = f"{out}/bot/weights/nn/rl/shards_v2"
    if not os.path.isdir(d):
        print("no shards dir for", w)
        continue
    n = len(os.listdir(d))
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        api.upload_folder(folder_path=d, path_in_repo="rl/shards_v2",
                          repo_id="amer224/territorial-bot-data",
                          repo_type="dataset", token=os.environ["HF_TOKEN"])
        print(f"uploaded {n} shards from worker {w}")
    except Exception as e:
        print(f"upload fail w{w}:", str(e)[:150])
print("MOVER_DONE")
