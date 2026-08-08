#!/usr/bin/env python3
"""Retry-push the CPU trainer + CPU v15 until a Kaggle CPU slot frees
(user is stopping GPU sessions / a worker finishes). Then exit."""
import subprocess
import sys
import time

done = {"trainer": False, "v15": False}
t0 = time.time()
while not all(done.values()) and time.time() - t0 < 4 * 3600:
    if not done["trainer"]:
        r = subprocess.run([sys.executable, "scripts/launch_loop_kernels.py",
                            "--trainer-only", "--trainer-delay-min", "5"],
                           capture_output=True, text=True)
        ok = "pushed rl-loop-trainer-cpu: Kernel push error" not in (r.stdout + r.stderr)
        done["trainer"] = ok
        print(f"[retry] trainer push {'OK' if ok else 'no slot yet'}", flush=True)
    if not done["v15"]:
        r = subprocess.run([sys.executable, "scripts/launch_train.py", "--push-only"],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        done["v15"] = ok
        print(f"[retry] v15 cpu push {'OK' if ok else 'no slot yet'}", flush=True)
    if not all(done.values()):
        time.sleep(300)
print(f"[retry] final: {done}", flush=True)
