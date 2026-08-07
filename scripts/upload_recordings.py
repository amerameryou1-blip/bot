#!/usr/bin/env python3
"""Upload recorded match sessions to Hugging Face.

Usage:
  python3 scripts/upload_recordings.py [recordings/session_xxx ...]

If no sessions are given, uploads EVERY session folder in recordings/.
Needs HF_TOKEN in env. Files land at:
  amer224/territorial-bot-data/recordings/<session_id>/
"""

import os
import sys
from pathlib import Path

REPO_ID = "amer224/territorial-bot-data"
RECORDINGS_ROOT = Path(__file__).resolve().parents[1] / "recordings"


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN env var required")
        return 1

    args = sys.argv[1:]
    sessions = [Path(a) for a in args] if args else sorted(
        [d for d in RECORDINGS_ROOT.iterdir() if d.is_dir()])

    if not sessions:
        print(f"no sessions found under {RECORDINGS_ROOT}")
        return 1

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    ok = True
    for s in sessions:
        sid = s.name
        n_frames = len(list((s / "frames").glob("*.jpg"))) if (s / "frames").is_dir() else 0
        print(f"uploading {sid} ({n_frames} frames)...", flush=True)
        try:
            api.upload_folder(
                folder_path=str(s),
                path_in_repo=f"recordings/{sid}",
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"real match recording {sid}",
            )
            print(f"  OK -> {REPO_ID}/recordings/{sid}", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            ok = False
    print("ALL UPLOADS DONE" if ok else "SOME UPLOADS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
