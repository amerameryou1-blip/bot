#!/usr/bin/env python3
"""Inject HF token into a TEMP copy of the GPU notebook, push to Kaggle, monitor.

The repo copy is token-free (public repo). This script only touches /tmp.
Usage: HF_TOKEN=... python3 scripts/launch_train.py [--rounds 100]
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "kaggle-push", "kaggle_train_nn.ipynb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", default=None, help="PPO_ROUNDS override")
    args = ap.parse_args()

    hf = os.environ.get("HF_TOKEN", "").strip()
    if not hf:
        print("FAIL: HF_TOKEN env required (injected into the private kernel copy)")
        sys.exit(1)

    tmp = tempfile.mkdtemp(prefix="train_launch_")
    nb = json.load(open(NB))
    # inject token into the env-fallback slots
    injected = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "\n".join(c["source"])
        if "os.environ.get('HF_TOKEN', '')" in src:
            src = src.replace("os.environ.get('HF_TOKEN', '')",
                              "os.environ.get('HF_TOKEN', '" + hf + "')")
            c["source"] = src.split("\n")
            injected += 1
    if args.rounds:
        for c in nb["cells"]:
            src = "\n".join(c["source"])
            if "PPO_ROUNDS', '100'" in src:
                c["source"] = src.replace("PPO_ROUNDS', '100'",
                                          "PPO_ROUNDS', '" + args.rounds + "'").split("\n")
    print("injected token into", injected, "cells")

    # metadata: script kernel
    meta = {
        "id": "amerameryou/bot-train-nn",
        "title": "bot-train-nn",
        "code_file": "kaggle_train_nn.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(tmp, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(tmp, "kaggle_train_nn.ipynb"), "w") as f:
        json.dump(nb, f)

    r = subprocess.run(["kaggle", "kernels", "push", "-p", tmp], capture_output=True, text=True)
    print(r.stdout.strip()[-400:])
    print(r.stderr.strip()[-400:] if r.returncode else "")
    if r.returncode != 0:
        sys.exit(1)
    print("PUSHED. Monitoring amerameryou/bot-train-nn...")
    for i in range(120):
        s = subprocess.run(["kaggle", "kernels", "status", "amerameryou/bot-train-nn"],
                           capture_output=True, text=True).stdout
        st = "running"
        if "COMPLETE" in s: st = "COMPLETE"
        elif "ERROR" in s: st = "ERROR"
        elif "QUEUED" in s: st = "QUEUED"
        elif "CANCEL" in s: st = "CANCELLED"
        print(f"[{i}] {st}")
        if st in ("COMPLETE", "ERROR", "CANCELLED"):
            out = os.path.join(tmp, "out")
            os.makedirs(out, exist_ok=True)
            subprocess.run(["kaggle", "kernels", "output", "amerameryou/bot-train-nn", "-p", out],
                           capture_output=True)
            logf = os.path.join(out, "bot-train-nn.log")
            if os.path.exists(logf):
                print("===== TRAIN LOG TAIL =====")
                try:
                    for d in json.load(open(logf)):
                        if d["stream_name"] == "stdout":
                            print(d["data"], end="")
                except Exception:
                    print(open(logf).read()[-4000:])
            sys.exit(0 if st == "COMPLETE" else 1)
        time.sleep(60)


if __name__ == "__main__":
    main()
