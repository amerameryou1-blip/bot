#!/usr/bin/env python3
"""Auto-label REAL game frames for vision fine-tuning.

Turns recordings (frames + meta with self/enemy colors) and mid-match
screenshots into per-pixel 5-class labels:
    0 water  1 neutral  2 me  3 enemy  4 ui
using the same color math as the live bot (calibration.py) — water hue band,
territory colors within tolerance, land tones, everything else = UI.

Also extracts CLICK targets from recordings (frame -> cell target + kind) for
behavior-cloning the click head on REAL clicks.

Output: <out>.npz with keys:
    rgb      (N,3,64,64) float32 0..1
    labels   (N,64,64)   int64    5 classes
    kind     (N,)        int64    click kind (0 expand,1 attack,2 bank)  [recordings]
    cell     (N,)        int64    16x16 target cell                      [recordings]
    pct      (N,)        float32  attack %                               [recordings]

Run:
  python3 scripts/label_real.py --recordings recordings --out weights/nn/real_vision.npz
  python3 scripts/label_real.py --images dir_of_pngs --out weights/nn/real_vision.npz
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

GRID = 16
SIZE = 64


def classify_frame(rgb, self_rgb, enemy_rgbs):
    """Return 5-class labels for an RGB uint8 frame.

    FIX (2026-08-08): the old version defaulted DARK pixels to UI and required
    water b>=45 / land total>=60. The real game map is dark-themed (dark navy
    ocean, dark land), so most map pixels were mislabeled UI — the NN could
    never learn water/land (observed: water acc 0.00, all-UI collapse).
    Now UI = KNOWN screen regions (leaderboard/banner/bottom bar) + bright
    low-chroma text; the whole map is classified by color with NO darkness
    floor and NEUTRAL as the final fallback.
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    H, W = r.shape

    # --- UI detection ---
    # Opaque UI: bottom bar, top banner, far-left edge. The leaderboard panel
    # is TRANSLUCENT (map shows through) — masking the whole rect would label
    # map-through pixels as UI (pollutes ui with blue). Instead we catch
    # leaderboard TEXT via the bright low-chroma rule below.
    ui = np.zeros((H, W), dtype=bool)
    ui[int(0.92 * H):, :] = True                         # bottom bar (opaque)
    ui[: max(2, int(0.05 * H)), :] = True                # top banner (opaque)
    ui[:, : max(2, int(0.015 * W))] = True               # far-left edge
    # bright low-chroma pixels anywhere = text/icons (leaderboard rows, % labels)
    total = r + g + b
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    ui |= (total >= 600) & (chroma < 24)

    out = np.full((H, W), 1, dtype=np.int64)  # default neutral (MAP land/fallback)

    # water: blue-dominant — NO brightness floor (dark navy ocean is water)
    water = (b > r + 10) & (b > g + 5) & (~ui)
    out[water] = 0

    # territory colors (me / enemy) within tolerance — take priority over
    # water, and only outside UI regions (UI swatches are not territory)
    sr, sg, sb = self_rgb
    me = (np.abs(r - sr) <= 36) & (np.abs(g - sg) <= 36) & (np.abs(b - sb) <= 36) & (~ui)
    out[me] = 2
    for er, eg, eb in enemy_rgbs:
        em = (np.abs(r - er) <= 36) & (np.abs(g - eg) <= 36) & (np.abs(b - eb) <= 36) & (~ui)
        out[em] = 3

    # land: green / beige / warm-gray (any brightness) — stays neutral (1)
    green = (g > r + 5) & (g > b + 5) & (g > 18) & (~ui)
    beige = (r > g + 4) & (g > b) & (r > 40) & (r < 250) & (g < 240) & (~ui)
    warmgray = (chroma >= 8) & (r >= g) & (g >= b) & (r > 40) & (~ui)
    # (neutral is already 1 — these masks just keep it 1; nothing to do)

    out[ui] = 4  # UI last — overwrites map classes inside UI regions
    return out


def resize_pair(rgb, labels, size=SIZE):
    """Resize frame + labels to SIZE. RGB stays uint8 (0..255) — float32
    lists OOM the sandbox at >3k frames; the trainer converts /255 on load."""
    rgb_s = np.array(Image.fromarray(rgb).resize((size, size), Image.BILINEAR)).astype(np.uint8)
    lab_s = np.array(Image.fromarray(labels.astype(np.uint8)).resize((size, size), Image.NEAREST)).astype(np.uint8)
    return rgb_s, lab_s


def load_frame(path):
    return np.asarray(Image.open(path).convert("RGB"))


# session ids (or prefixes) curated out by the vision agent after looking at
# the frames: broken-camera or modal-frozen matches pollute click labels.
SKIP_SESSIONS: set = set()


def process_recordings(rec_root: Path) -> dict:
    rgb_list, lab_list, kind_list, cell_list, pct_list = [], [], [], [], []
    sessions = sorted(d for d in rec_root.iterdir() if d.is_dir())
    if not sessions:
        print(f"no sessions under {rec_root}")
        return None
    for sess in sessions:
        meta_path = sess / "meta.json"
        if not meta_path.exists():
            print(f"  skip {sess.name}: no meta.json")
            continue
        meta = json.loads(meta_path.read_text())
        if sess.name in SKIP_SESSIONS or any(sess.name.startswith(s)
                                             for s in SKIP_SESSIONS if s):
            print(f"  skip {sess.name}: curated out (bad camera / modal freeze)")
            continue
        if not meta.get("self_color") or not meta.get("enemy_colors"):
            print(f"  skip {sess.name}: no self/enemy colors in meta")
            continue
        self_rgb = tuple(meta["self_color"])
        enemy_rgbs = [tuple(c) for c in meta.get("enemy_colors", [])]
        frames = sorted((sess / "frames").glob("*.jpg"))
        clicks = {}
        cpath = sess / "clicks.jsonl"
        if cpath.exists():
            for line in cpath.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                c = json.loads(line)
                clicks[int(c["frame"])] = c
        n = 0
        for f in frames:
            idx = int(f.stem)
            try:
                rgb = load_frame(f)
            except Exception:
                continue
            labels = classify_frame(rgb, self_rgb, enemy_rgbs)
            rgb_s, lab_s = resize_pair(rgb, labels)
            rgb_list.append(rgb_s); lab_list.append(lab_s)
            if idx in clicks:
                c = clicks[idx]
                # clicks are in ORIGINAL screen space; frames may be downscaled
                o_w, o_h = meta.get("frame_orig_size", [labels.shape[1], labels.shape[0]])
                cy = int(c["y"] / o_h * GRID)
                cx = int(c["x"] / o_w * GRID)
                cy = min(GRID - 1, max(0, cy)); cx = min(GRID - 1, max(0, cx))
                kind_map = {"expand": 0, "attack": 1, "bank": 2}
                kind_list.append(kind_map.get(c["kind"], 0))
                cell_list.append(cy * GRID + cx)
                pct_list.append(float(c.get("pct") or 0.0))
            n += 1
        print(f"  {sess.name}: {n} frames labeled")
    if not rgb_list:
        print("no labeled frames produced")
        return None
    return {
        "rgb": np.array(rgb_list), "labels": np.array(lab_list),
        "kind": np.array(kind_list, dtype=np.int64) if kind_list else None,
        "cell": np.array(cell_list, dtype=np.int64) if cell_list else None,
        "pct": np.array(pct_list, dtype=np.float32) if pct_list else None,
    }


def process_images(img_root: Path) -> dict:
    rgb_list, lab_list = [], [], []
    files = sorted(p for p in img_root.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        print(f"no images under {img_root}")
        return None
    for f in files:
        rgb = load_frame(f)
        # start-phase screenshots have no territories: label with no territory colors
        labels = classify_frame(rgb, (0, 0, 0), [])
        rgb_s, lab_s = resize_pair(rgb, labels)
        rgb_list.append(rgb_s); lab_list.append(lab_s)
    print(f"{len(rgb_list)} frames labeled (start-phase screenshots)")
    return {"rgb": np.array(rgb_list), "labels": np.array(lab_list),
            "kind": None, "cell": None, "pct": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", type=str, help="folder of recording sessions")
    ap.add_argument("--images", type=str, help="folder of screenshots")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--save-anyway", action="store_true",
                    help="save even if no click labels (vision-only dataset)")
    ap.add_argument("--skip", type=str, default="",
                    help="comma-separated session id prefixes to exclude")
    args = ap.parse_args()
    global SKIP_SESSIONS
    SKIP_SESSIONS = set(s.strip() for s in args.skip.split(",") if s.strip())

    if args.recordings:
        data = process_recordings(Path(args.recordings))
    elif args.images:
        data = process_images(Path(args.images))
    else:
        print("need --recordings or --images")
        return 1
    if data is None:
        return 1
    if data["kind"] is None and not args.save_anyway:
        print("no click labels — pass --save-anyway for a vision-only dataset")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # store rgb as uint8 (0..255) and labels as uint8 — ~6x smaller npz;
    # train_nn.py converts back to float on load.
    rgb_arr = data["rgb"] if data["rgb"].dtype == np.uint8 else (data["rgb"] * 255.0).astype(np.uint8)
    np.savez_compressed(out,
                        rgb=rgb_arr,
                        labels=data["labels"].astype(np.uint8),
                        kind=data["kind"] if data["kind"] is not None else np.zeros(0, dtype=np.int64),
                        cell=data["cell"] if data["cell"] is not None else np.zeros(0, dtype=np.int64),
                        pct=data["pct"] if data["pct"] is not None else np.zeros(0, dtype=np.float32),
                        dtype_comment="rgb uint8 0..255, labels uint8; convert /255 on load")
    n = len(data["rgb"])
    clicks = len(data["kind"]) if data["kind"] is not None else 0
    print(f"saved {out}: {n} frames, {clicks} click labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
