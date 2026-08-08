#!/usr/bin/env python3
"""Push NEW recording sessions to HF via the private-GitHub transfer route.

1. zip sessions not yet uploaded (tracked in recordings/.pushed.log)
2. push zip to amerameryou1-blip/bot-recordings (private)
3. trigger the migrate kernel (clones repo -> uploads every zip to HF)
4. poll kernel; on COMPLETE, delete the uploaded sessions locally (workspace budget)

Usage: HF_TOKEN=... GH_TOKEN=... python3 scripts/push_github.py [--keep]
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "recordings")
PUSHED_LOG = os.path.join(REC, ".pushed.log")
GH_REPO = "https://{tok}@github.com/amerameryou1-blip/bot-recordings.git"
KERNEL = "amerameryou/migrate-recordings-b3"


def pushed_ids() -> set:
    if os.path.exists(PUSHED_LOG):
        return set(l.strip() for l in open(PUSHED_LOG) if l.strip())
    return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep local copies after upload")
    args = ap.parse_args()

    gh = os.environ.get("GH_TOKEN", "")
    hf = os.environ.get("HF_TOKEN", "")
    if not (gh and hf):
        print("FAIL: GH_TOKEN + HF_TOKEN required"); sys.exit(1)

    sessions = []
    for meta in sorted(glob.glob(os.path.join(REC, "*/meta.json"))):
        sid = json.load(open(meta)).get("session_id", os.path.basename(os.path.dirname(meta)))
        if sid in pushed_ids():
            continue
        nf = json.load(open(meta)).get("frames", 0)
        if nf == 0:
            print(f"  skip empty {sid}"); continue
        sessions.append(os.path.dirname(meta))
    if not sessions:
        print("nothing new to push")
        return 0
    print(f"new sessions: {len(sessions)}")

    # 1) zip new sessions
    tmp = tempfile.mkdtemp(prefix="pushgh_")
    zip_path = os.path.join(tmp, f"recordings_{time.strftime('%Y%m%d-%H%M')}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for sess in sessions:
            for root, _, files in os.walk(sess):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.join("recordings", os.path.relpath(full, REC))
                    z.write(full, rel)
    print(f"zip: {os.path.basename(zip_path)} ({os.path.getsize(zip_path)/1e6:.1f} MB)")

    # 2) push to private repo
    repo = os.path.join(tmp, "bot-recordings")
    subprocess.run(["git", "clone", "-q", GH_REPO.format(tok=gh), repo], check=True)
    shutil.copy(zip_path, os.path.join(repo, os.path.basename(zip_path)))
    subprocess.run(["git", "-C", repo, "add", os.path.basename(zip_path)], check=True)
    subprocess.run(["git", "-C", repo, "-c", "user.email=bot@users.noreply.github.com",
                    "-c", "user.name=bot", "commit", "-q", "-m", "recordings batch"], check=True)
    r = subprocess.run(["git", "-C", repo, "push", "-q"], capture_output=True, text=True)
    if r.returncode != 0:
        print("push failed:", r.stderr[-400:]); sys.exit(1)
    print("pushed to private repo")

    # 3) trigger migrate kernel (it clones the repo and uploads everything to HF)
    #    tokens injected into a TEMP copy (repo copy is token-free)
    ktmp = os.path.join(tmp, "kernel")
    shutil.copytree(os.path.join(ROOT, "kaggle-push", "migrate_kernel"), ktmp)
    km = os.path.join(ktmp, "migrate.py")
    ksrc = open(km).read()
    ksrc = ksrc.replace('os.environ.get("HF_TOKEN", "")', 'os.environ.get("HF_TOKEN", "' + hf + '")')
    ksrc = ksrc.replace('os.environ.get("GH_TOKEN", "")', 'os.environ.get("GH_TOKEN", "' + gh + '")')
    open(km, "w").write(ksrc)
    r = subprocess.run(["kaggle", "kernels", "push", "-p", ktmp],
                       capture_output=True, text=True)
    print(r.stdout.strip()[-200:] or r.stderr.strip()[-200:])
    if r.returncode != 0:
        print("kernel push failed"); sys.exit(1)

    # 4) poll
    for i in range(60):
        s = subprocess.run(["kaggle", "kernels", "status", KERNEL],
                           capture_output=True, text=True).stdout
        st = "running"
        if "COMPLETE" in s: st = "COMPLETE"
        elif "ERROR" in s: st = "ERROR"
        print(f"[{i}] {st}")
        if st in ("COMPLETE", "ERROR"):
            break
        time.sleep(20)
    if st != "COMPLETE":
        print("FAIL: migrate kernel did not complete"); sys.exit(1)

    # 5) mark pushed + delete local copies (unless --keep)
    with open(PUSHED_LOG, "a") as f:
        for sess in sessions:
            f.write(os.path.basename(sess) + "\n")
    if not args.keep:
        for sess in sessions:
            shutil.rmtree(sess, ignore_errors=True)
        print(f"deleted {len(sessions)} local sessions (workspace budget)")
    print(f"DONE: {len(sessions)} sessions uploaded to HF")
    return 0


if __name__ == "__main__":
    sys.exit(main())
