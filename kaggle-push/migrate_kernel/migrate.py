import glob, json, os, subprocess, sys, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
from huggingface_hub import HfApi
TOKEN = os.environ.get("HF_TOKEN", "")
GH = os.environ.get("GH_TOKEN", "")
api = HfApi(token=TOKEN)
r = subprocess.run(["git", "clone", "-q", f"https://{GH}@github.com/amerameryou1-blip/bot-recordings.git", "/kaggle/working/recs"], capture_output=True, text=True)
print("clone rc:", r.returncode)
zips = glob.glob("/kaggle/working/recs/**/*.zip", recursive=True)
print("zips:", [os.path.basename(z) for z in zips])
if not zips:
    print("FAIL: no zips"); sys.exit(1)
ok = 0; total = 0
for zf in zips:
    dest = "/kaggle/working/x_" + os.path.basename(zf).replace(".zip","")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zf) as z:
        z.extractall(dest)
    sessions = sorted(glob.glob(os.path.join(dest, "**", "meta.json"), recursive=True))
    print(f"{os.path.basename(zf)}: {len(sessions)} sessions")
    for meta_path in sessions:
        sess = os.path.dirname(meta_path); sid = os.path.basename(sess)
        meta = json.load(open(meta_path)); nf = meta.get("frames", 0)
        if nf == 0:
            print(f"[SKIP] {sid}: 0 frames"); continue
        try:
            api.upload_folder(folder_path=sess, path_in_repo=f"recordings/{sid}",
                              repo_id="amer224/territorial-bot-data", repo_type="dataset",
                              commit_message=f"real match {sid} ({nf} frames)")
            ok += 1; total += nf
        except Exception as e:
            print(f"[FAIL] {sid}: {e}")
print(f"SUMMARY: {ok} sessions, {total} frames uploaded to HF")
sys.exit(0 if ok else 1)
