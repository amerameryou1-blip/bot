"""Real-match recorder — frames + clicks + metadata for vision/click training.

Every decision tick records:
  - the frame (JPEG) the bot SAW,
  - the action it CHOSE (kind, target x/y, troop %, reason).
That is exactly the supervised data the click-head needs: "given this frame,
click HERE". The auto-labeler (scripts/label_real.py) later derives per-pixel
me/enemy/water/UI labels from the frames + leaderboard OCR.

Session layout:
  recordings/<session_id>/
    meta.json     map name (if OCR'd), self color, enemy colors, timestamps
    frames/000000.jpg ...
    clicks.jsonl  {"t":..., "x":..., "y":..., "kind":..., "pct":..., "frame":N}
    report.json   battle report at match end

Upload: set HF_TOKEN and call `finish(upload=True)` — pushes the whole session
to HF dataset `amer224/territorial-bot-data` under recordings/<session_id>/.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

JPEG_QUALITY = 82
DEFAULT_ROOT = Path("recordings")
HF_REPO = "amer224/territorial-bot-data"


class GameRecorder:
    def __init__(self, root: str | Path = DEFAULT_ROOT, session_id: str | None = None):
        self.root = Path(root)
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.dir = self.root / self.session_id
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._frame_idx = 0
        self._clicks: list[dict] = []
        self._meta: dict = {
            "session_id": self.session_id,
            "started_at": time.time(),
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._wrote_meta = False
        self._closed = False

    # -- meta ---------------------------------------------------------------

    def update_meta(self, **fields) -> None:
        self._meta.update(fields)
        self._write_meta()

    def _write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self._meta, indent=2))
        self._wrote_meta = True

    # -- frames -------------------------------------------------------------

    def record_frame(self, frame: np.ndarray, t: float | None = None) -> int:
        """Save one frame, return its index."""
        if self._closed:
            return self._frame_idx - 1
        idx = self._frame_idx
        self._frame_idx += 1
        try:
            im = Image.fromarray(frame[..., :3].astype(np.uint8))
            im.save(self.frames_dir / f"{idx:06d}.jpg", quality=JPEG_QUALITY)
            # occasionally flush click log so a crash never loses data
            if idx % 50 == 0:
                self._flush_clicks()
        except Exception as e:
            print(f"[recorder] frame {idx} save error: {e}", flush=True)
        return idx

    # -- clicks -------------------------------------------------------------

    def record_click(self, t: float, x: int, y: int, kind: str,
                     pct: float | None = None, frame: int = -1,
                     reason: str = "") -> None:
        if self._closed:
            return
        self._clicks.append({
            "t": round(t, 3),
            "x": int(x), "y": int(y),
            "kind": kind,
            "pct": None if pct is None else round(float(pct), 3),
            "frame": int(frame),
            "reason": reason,
        })
        self._flush_clicks()  # append per line, crash-safe

    def _flush_clicks(self) -> None:
        if not self._clicks:
            return
        with open(self.dir / "clicks.jsonl", "a") as f:
            for c in self._clicks:
                f.write(json.dumps(c) + "\n")
        self._clicks.clear()

    # -- finish + upload ----------------------------------------------------

    def finish(self, report: dict | None = None, upload: bool = False,
               hf_token: str = "") -> Path:
        self._closed = True
        self._flush_clicks()
        self._meta["ended_at"] = time.time()
        self._meta["frames"] = self._frame_idx
        self._meta["clicks"] = self._frames_to_clicks_count()
        if report:
            self._meta["report"] = report
        if report and "last_survivor" in report:
            self._meta["last_survivor"] = bool(report["last_survivor"])
        self._write_meta()

        (self.dir / "report.json").write_text(json.dumps(report or {}, indent=2))

        if upload:
            self.upload(hf_token)
        print(f"[recorder] session {self.session_id}: {self._frame_idx} frames, "
              f"{self._meta.get('clicks', 0)} clicks at {self.dir}", flush=True)
        return self.dir

    def _frames_to_clicks_count(self) -> int:
        n = 0
        try:
            for _ in open(self.dir / "clicks.jsonl"):
                n += 1
        except FileNotFoundError:
            pass
        return n

    def upload(self, hf_token: str) -> None:
        """Upload the session folder to HF dataset recordings/<session_id>/."""
        if not hf_token:
            print("[recorder] no HF_TOKEN set — session left locally at "
                  f"{self.dir}. To upload later: python3 scripts/upload_recordings.py {self.dir}",
                  flush=True)
            return
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            info = api.upload_folder(
                folder_path=str(self.dir),
                path_in_repo=f"recordings/{self.session_id}",
                repo_id=HF_REPO,
                repo_type="dataset",
                commit_message=f"real match recording {self.session_id}",
            )
            print(f"[recorder] uploaded {len(info.uploaded_files) if hasattr(info,'uploaded_files') else '?'} "
                  f"files to {HF_REPO}/recordings/{self.session_id}", flush=True)
        except Exception as e:
            print(f"[recorder] UPLOAD FAILED: {e} — session kept locally at {self.dir}", flush=True)


def make_recorder(root: str | Path = DEFAULT_ROOT) -> GameRecorder | None:
    return GameRecorder(root=root)
