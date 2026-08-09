#!/usr/bin/env python3
"""Emergency data route (sprint 2026-08-09): HF commit budget was saturated,
so v2 shards go to the private GH repo bot-recordings instead. The GPU kernel
clones it. Consumes local shards after pushing (workspace budget)."""
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "weights/nn/rl/shards_v2"
GH = os.environ.get("GH_URL", "")
CLONE = "/tmp/shards_gh"

def main():
    subprocess.run(["git", "clone", "-q", GH, CLONE], check=True)
    end = time.time() + 6.5 * 3600
    pushed = 0
    while time.time() < end:
        new = sorted(SRC.glob("shard_v2_*.npz"))
        if new:
            for f in new:
                subprocess.run(["cp", str(f), f"{CLONE}/{f.name}"])
            subprocess.run(["git", "-C", CLONE, "add", "-A"], check=True)
            subprocess.run(["git", "-C", CLONE, "-c", "user.email=b@b.c",
                            "-c", "user.name=bot", "commit", "-q",
                            "-m", f"shards +{len(new)}"], check=True)
            r = subprocess.run(["git", "-C", CLONE, "push", "-q"],
                               capture_output=True)
            if r.returncode == 0:
                pushed += len(new)
                for f in new:
                    f.unlink()
                print(f"[gh] pushed {len(new)} (total {pushed})", flush=True)
            else:
                print("[gh] push failed, retry later", flush=True)
        time.sleep(120)
    print(f"[gh] done, pushed {pushed}", flush=True)

if __name__ == "__main__":
    main()
