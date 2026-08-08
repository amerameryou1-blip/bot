#!/usr/bin/env python3
"""HF dataset hygiene (user request 2026-08-08: "delete the rubbish").

Commands:
  report            — print per-folder sizes (GB) and file counts
  prune-old         — delete OLD procedural filler shards at repo root
                      (3-bot game5 data, superseded by real-map rl shards)
  prune-shards N    — keep only the N newest rl/shards/*, delete the rest
  prune-recordings LIST — delete recordings/<sid> folders for comma-separated
                      session ids (bad-camera / modal-frozen matches)

Dry-run by default; add --apply to really delete.
Usage: HF_TOKEN=... python3 scripts/hf_cleanup.py <cmd> [args] [--apply]
"""
import os
import sys
import json
import urllib.request
from collections import defaultdict

TOKEN = os.environ.get("HF_TOKEN", "").strip()
REPO = "amer224/territorial-bot-data"
OLD_ROOT_SHARDS = ("shard_desert_medium_s301-340.npz", "shard_island_medium_s1-3.npz",
                   "shard_island_medium_s101-140.npz", "shard_lakes_medium_s1-3.npz",
                   "shard_mountains_medium_s201-240.npz", "shard_swamp_medium_s401-440.npz")


def api(path):
    req = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{REPO}/{path}",
        headers={"Authorization": f"Bearer {TOKEN}"})
    return json.load(urllib.request.urlopen(req))


def all_files():
    out, cursor = [], ""
    while True:
        url = f"tree/main?recursive=true&limit=1000" + (f"&cursor={cursor}" if cursor else "")
        page = api(url)
        if not page:
            break
        out += [f for f in page if f.get("type") == "file"]
        if len(page) < 1000:
            break
        import base64
        cursor = base64.b64encode(json.dumps([page[-1]["path"]]).encode()).decode()
    return out


def main():
    args = [a for a in sys.argv[1:] if a != "--apply"]
    apply = "--apply" in sys.argv[1:]
    if not TOKEN:
        print("HF_TOKEN required"); sys.exit(1)
    cmd = args[0] if args else "report"

    if cmd == "report":
        fs = all_files()
        tot = sum(f.get("size", 0) for f in fs)
        print(f"TOTAL {tot/1e9:.2f} GB, {len(fs)} files")
        by = defaultdict(lambda: [0, 0])
        for f in fs:
            top = f["path"].split("/")[0] if "/" in f["path"] else "(root)"
            by[top][0] += 1
            by[top][1] += f.get("size", 0)
        for k, (n, s) in sorted(by.items(), key=lambda kv: -kv[1][1]):
            print(f"  {k:20s} {n:5d} files {s/1e9:7.2f} GB")
        return

    from huggingface_hub import HfApi, DeleteFiles
    api_h = HfApi(token=TOKEN)
    targets = []
    if cmd == "prune-old":
        targets = list(OLD_ROOT_SHARDS)
    elif cmd == "prune-shards":
        keep = int(args[1])
        fs = [f for f in all_files() if f["path"].startswith("rl/shards/")]
        fs.sort(key=lambda f: f["path"])
        targets = [f["path"] for f in fs[:-keep]] if len(fs) > keep else []
        print(f"keeping {min(keep, len(fs))} newest of {len(fs)}")
    elif cmd == "prune-recordings":
        for sid in args[1].split(","):
            sid = sid.strip()
            if sid:
                targets += [f["path"] for f in all_files()
                            if f["path"].startswith(f"recordings/{sid}/")]
    else:
        print("unknown cmd"); sys.exit(1)
    for p in targets:
        print(f"  {'DELETE' if apply else 'would delete'} {p}", flush=True)
    if apply and targets:
        # single commit for ALL deletes (HF commit budget: 128/hour)
        api_h.create_commit(repo_id=REPO, repo_type="dataset", token=TOKEN,
                            commit_message=f"cleanup: {len(targets)} files",
                            operations=[DeleteFiles(path_in_repo=p) for p in targets])
    print("done" if apply else "DRY RUN — pass --apply to delete")


if __name__ == "__main__":
    main()
