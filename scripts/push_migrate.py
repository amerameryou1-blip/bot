#!/usr/bin/env python3
"""Push ONLY the recordings->HF migrate kernel (tokens injected at push).

Use when a CPU slot frees up and zips are waiting on the private GH repo
(amerameryou1-blip/bot-recordings) but locals were already cleaned.

Usage: HF_TOKEN=... GH_TOKEN=... KAGGLE_CONFIG_DIR=... python3 scripts/push_migrate.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "amerameryou/migrate-recordings-b3"


def main():
    hf = os.environ.get("HF_TOKEN", "").strip()
    gh = os.environ.get("GH_TOKEN", "").strip()
    if not (hf and gh):
        print("HF_TOKEN + GH_TOKEN required"); sys.exit(1)
    ktmp = tempfile.mkdtemp(prefix="migrate_")
    shutil.copytree(ROOT / "kaggle-push" / "migrate_kernel", ktmp, dirs_exist_ok=True)
    km = Path(ktmp) / "migrate.py"
    src = km.read_text()
    src = src.replace('os.environ.get("HF_TOKEN", "")',
                      f'os.environ.get("HF_TOKEN", "{hf}")')
    src = src.replace('os.environ.get("GH_TOKEN", "")',
                      f'os.environ.get("GH_TOKEN", "{gh}")')
    km.write_text(src)
    r = subprocess.run(["kaggle", "kernels", "push", "-p", ktmp],
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-200:])
    if r.returncode != 0:
        print("FAIL: push rejected (CPU slots full?)"); sys.exit(1)
    seen = False
    for i in range(90):
        s = subprocess.run(["kaggle", "kernels", "status", KERNEL],
                           capture_output=True, text=True).stdout
        if "RUNNING" in s or "QUEUED" in s:
            seen = True
        elif seen and ("COMPLETE" in s or "ERROR" in s):
            print(f"[{i}] done: {s.strip()[-80:]}")
            sys.exit(0 if "COMPLETE" in s else 1)
        time.sleep(20)
    print("timeout waiting for migrate"); sys.exit(1)


if __name__ == "__main__":
    main()
