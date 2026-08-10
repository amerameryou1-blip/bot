#!/usr/bin/env python3
"""Sprint stage 3: when the GPU-pretrained teacher appears on HF,
download it and run live win attempts with live_v2.py (up to 4 matches)."""
import os
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def teacher_url():
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN", ""))
        fs = api.list_repo_files("amer224/territorial-bot-data",
                                 repo_type="dataset",
                                 token=os.environ.get("HF_TOKEN", ""))
        if any(f == "v2/teacher.pt" for f in fs):
            return "hf"
    except Exception as e:
        print("[live] teacher check err:", str(e)[:80], flush=True)
    return None


def main():
    end = time.time() + 7 * 3600
    while time.time() < end:
        url = teacher_url()
        if url:
            from huggingface_hub import hf_hub_download
            dst = os.path.join(REPO, "weights/nn/v2/teacher.pt")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            got = hf_hub_download("amer224/territorial-bot-data", "v2/teacher.pt",
                                  repo_type="dataset",
                                  token=os.environ.get("HF_TOKEN", ""))
            import shutil
            shutil.copy(got, dst)
            print("[live] teacher downloaded, starting attempts", flush=True)
            for i in range(4):
                env = dict(os.environ)
                env.update({"BRAIN_PT": dst, "ARCH": "teacher",
                            "PLAY_MINUTES": "5", "PYTHONPATH":
                            os.path.join(REPO, "src")})
                r = subprocess.run([sys.executable,
                                    os.path.join(REPO, "scripts/live_v2.py")],
                                   env=env, capture_output=True, text=True,
                                   timeout=600)
                print(f"[live] attempt {i}: rc={r.returncode}",
                      r.stdout[-300:], flush=True)
                time.sleep(20)
            print("[live] all attempts done", flush=True)
            return
        time.sleep(300)
    print("[live] window ended, no teacher", flush=True)


if __name__ == "__main__":
    main()
