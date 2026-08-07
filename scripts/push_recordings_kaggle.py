#!/usr/bin/env python3
"""Zip recordings -> private Kaggle dataset -> push migration kernel -> monitor.

Run: HF_TOKEN via env (only used by the KAGGLE-side kernel; this script itself
only needs the Kaggle token via KAGGLE_CONFIG_DIR / KAGGLE_API_TOKEN).

Usage: python3 scripts/push_recordings_kaggle.py --tag b1
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "recordings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b1")
    ap.add_argument("--dataset", default=None, help="kaggle dataset slug (default auto)")
    ap.add_argument("--kernel", default=None)
    args = ap.parse_args()

    sessions = sorted(glob.glob(os.path.join(REC, "*/meta.json")))
    if not sessions:
        print("FAIL: no recordings found"); sys.exit(1)
    nframes = 0; nclicks = 0
    kept = []
    for m in sessions:
        meta = json.load(open(m))
        if meta.get("frames", 0) == 0:
            print(f"  skip empty session {meta.get('session_id')}")
            continue
        nframes += meta.get("frames", 0); nclicks += meta.get("clicks", 0)
        kept.append(m)
    sessions = kept
    if not sessions:
        print("FAIL: no usable recordings"); sys.exit(1)
    print(f"sessions={len(sessions)} frames={nframes} clicks={nclicks}")

    # 1) zip
    tmp = tempfile.mkdtemp(prefix="recs_kaggle_")
    zip_path = os.path.join(tmp, f"recordings_{args.tag}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for sess in glob.glob(os.path.join(REC, "*")):
            for root, _, files in os.walk(sess):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, REC)
                    z.write(full, rel)
    print(f"zip: {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")

    # 2) kaggle dataset
    ds_dir = os.path.join(tmp, "ds")
    os.makedirs(ds_dir, exist_ok=True)
    shutil.copy(zip_path, os.path.join(ds_dir, os.path.basename(zip_path)))
    slug = args.dataset or f"territorial-bot-recordings-{args.tag}"
    with open(os.path.join(ds_dir, "dataset-metadata.json"), "w") as f:
        json.dump({
            "id": f"amerameryou/{slug}",
            "title": f"territorial bot recordings {args.tag}",
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        }, f)
    r = subprocess.run(["kaggle", "datasets", "create", "-p", ds_dir, "-q"],
                       capture_output=True, text=True)
    print("dataset create:", r.stdout.strip()[-300:], r.stderr.strip()[-200:])

    # 3) migration kernel
    kdir = os.path.join(tmp, "kernel")
    os.makedirs(kdir, exist_ok=True)
    kernel = args.kernel or f"migrate-recordings-{args.tag}"
    with open(os.path.join(kdir, "kernel-metadata.json"), "w") as f:
        json.dump({
            "id": f"amerameryou/{kernel}",
            "title": kernel,
            "code_file": "migrate.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": True,
            "dataset_sources": [f"amerameryou/{slug}"],
            "competition_sources": [],
            "kernel_sources": [],
        }, f, indent=1)
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("FAIL: set HF_TOKEN env (used inside the migration kernel to upload to HF)")
        sys.exit(1)
    with open(os.path.join(kdir, "migrate.py"), "w") as f:
        f.write(f'''import glob, json, os, subprocess, sys, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
from huggingface_hub import HfApi
TOKEN = "{hf_token}"
REPO = "amer224/territorial-bot-data"
api = HfApi(token=TOKEN)
INPUT = "/kaggle/input/{slug}"
zips = glob.glob(os.path.join(INPUT, "**", "*.zip"), recursive=True)
print("zips:", zips)
if not zips:
    print("FAIL: no zip in input"); sys.exit(1)
dest = "/kaggle/working/recordings"
os.makedirs(dest, exist_ok=True)
with zipfile.ZipFile(zips[0]) as z:
    z.extractall(dest)
sessions = sorted(glob.glob(os.path.join(dest, "*/meta.json")))
print("sessions:", len(sessions))
ok = 0
for meta_path in sessions:
    sess = os.path.dirname(meta_path); sid = os.path.basename(sess)
    meta = json.load(open(meta_path)); nf = meta.get("frames", 0)
    try:
        info = api.upload_folder(folder_path=sess, path_in_repo=f"recordings/{{sid}}",
                                 repo_id=REPO, repo_type="dataset",
                                 commit_message=f"real match {{sid}} ({{nf}} frames)")
        print(f"[PASS] {{sid}}: {{nf}} frames uploaded")
        ok += 1
    except Exception as e:
        print(f"[FAIL] {{sid}}: {{e}}")
print(f"SUMMARY: {{ok}}/{{len(sessions)}} sessions uploaded")
sys.exit(0 if ok == len(sessions) else 1)
''')
    r = subprocess.run(["kaggle", "kernels", "push", "-p", kdir], capture_output=True, text=True)
    print("kernel push:", r.stdout.strip()[-300:], r.stderr.strip()[-200:])
    print(f"KERNEL: https://www.kaggle.com/code/amerameryou/{kernel}")


if __name__ == "__main__":
    main()
