#!/usr/bin/env python3
"""Dual-account fleet watchdog (2026-08-14, user order).

Account 1 (amerameryou): 4 workers + trainer (+w5 spare).
Account 2 (amer38, user-created for extra CPU quota): same shape.
Tokens come ONLY from env KG1 / KG2 — never stored in this file.

Every cycle: status both fleets, relaunch dead kernels, push trainers
best-effort, pre-fetch unaudited HF shards for the agent's review.
"""
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_loop_kernels as L

FLEETS = [
    {"owner": "amerameryou", "tok": os.environ.get("KG1", ""),
     "workers": [f"rl-v3-worker-{i}" for i in range(1, 6)],
     "trainer": "rl-loop-trainer-cpu"},
    # account 2 = pure data (5 workers). ONE trainer total for the whole
    # project — a second trainer would race on best.json for nothing.
    # (rl-b-trainer v1 runs out its 9h once; we do NOT relaunch it.)
    {"owner": "amer38", "tok": os.environ.get("KG2", ""),
     "workers": [f"rl-b-worker-{i}" for i in range(1, 6)],
     "trainer": ""},
]
ALIVE = ("RUNNING", "QUEUED")


def status(owner, slug, tok):
    env = dict(os.environ)
    if tok:
        env["KAGGLE_API_TOKEN"] = tok
    r = subprocess.run(["kaggle", "kernels", status_arg(owner, slug)],
                       capture_output=True, text=True, env=env)
    out = (r.stdout + r.stderr)
    for t in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "RUNNING", "QUEUED"):
        if t in out:
            return t
    return out.strip()[:60]


def status_arg(owner, slug):
    return ["kernels", "status", f"{owner}/{slug}"]


def worker_code(hf):
    return (L.HD_WORKER_BOOT.replace("@@HF@@", hf)
            .replace("@@REPO@@", L.GH_REPO_URL)
            .replace("@@HOURS@@", "8.5"))


def trainer_code(hf):
    return (L.TRAINER_BOOT.replace("@@HF@@", hf)
            .replace("@@REPO@@", L.GH_REPO_URL)
            .replace("@@HOURS@@", "9.0").replace("@@DELAY@@", "0"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--relaunch", action="store_true")
    a = ap.parse_args()
    hf = os.environ.get("HF_TOKEN", "").strip()

    for fl in FLEETS:
        if not fl["tok"]:
            print(f"{fl['owner']}: no token in env — skip", flush=True)
            continue
        # fleet order: 4 workers + trainer get slots; w5 is the spare
        tr = [fl["trainer"]] if fl["trainer"] else []
        order = fl["workers"][:4] + tr + fl["workers"][4:]
        dead = []
        for k in order:
            s = status(fl["owner"], k, fl["tok"])
            print(f"{fl['owner']}/{k:22s} {s}", flush=True)
            if not any(x in s for x in ALIVE):
                dead.append(k)
        if a.relaunch and dead and hf:
            for k in dead:
                if k == fl["trainer"]:
                    code = trainer_code(hf)
                else:
                    code = worker_code(hf)
                out = L.push_kernel(k, k, code, owner=fl["owner"],
                                    token=fl["tok"])
                print(f"relaunched {fl['owner']}/{k}: {out[:120]}", flush=True)

    # rival-brain upload detector (2026-08-14): the other agent will drop his
    # model under brains/ — flag it loudly the moment it appears.
    try:
        from huggingface_hub import HfApi
        hf = os.environ.get("HF_TOKEN", "").strip()
        if hf:
            fs = [f for f in HfApi(token=hf).list_repo_files(
                "amer224/territorial-bot-data", repo_type="dataset", token=hf)
                if f.startswith("brains/")]
            prev = -1
            try:
                prev = int(open("/tmp/rival_brains_count").read())
            except Exception:
                pass
            if len(fs) != prev:
                print(f"RIVAL BRAINS/: {len(fs)} files "
                      f"({'NEW UPLOAD' if len(fs) > max(prev,0) else 'changed'}) "
                      f"-> review them!", flush=True)
            open("/tmp/rival_brains_count", "w").write(str(len(fs)))
    except Exception as e:
        print(f"rival check skipped: {e}", flush=True)

    # pre-fetch unaudited HF shards for the agent's morning review
    try:
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "review_v2.py"), "--fetch"], timeout=900)
    except Exception as e:
        print(f"review fetch skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
