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
    """Return 5-class labels for an RGB uint8 frame."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    H, W = r.shape
    out = np.full((H, W), 4, dtype=np.int64)  # default UI

    # water: blue-dominant (dark navy to bright blue)
    water = (b > r + 10) & (b > g + 5) & (b >= 45)
    out[water] = 0

    # territory colors (me / enemy) within tolerance
    me = np.ones((H, W), dtype=bool)
    sr, sg, sb = self_rgb
    me &= (np.abs(r - sr) <= 36) & (np.abs(g - sg) <= 36) & (np.abs(b - sb) <= 36)
    out[me] = 2
    for er, eg, eb in enemy_rgbs:
        em = (np.abs(r - er) <= 36) & (np.abs(g - eg) <= 36) & (np.abs(b - eb) <= 36)
        out[em] = 3

    # land: green / beige / warm-gray / white-tile
    total = r + g + b
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    green = (g > r + 5) & (g > b + 5) & (g > 30) & (total >= 60)
    beige = (r > g + 4) & (g > b) & (r > 90) & (r < 245) & (g < 235)
    warmgray = (chroma >= 10) & (r >= g) & (g >= b) & (r > 90)
    light = (total >= 640) & (chroma < 16)
    land = (green | beige | warmgray | light) & (out == 4)
    out[land] = 1
    return out


def resize_pair(rgb, labels, size=SIZE):
    rgb_s = np.array(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))
    lab_s = np.array(Image.fromarray(labels.astype(np.uint8)).resize((size, size), Image.NEAREST))
    return rgb_s.astype(np.float32) / 255.0, lab_s.astype(np.int64)


def load_frame(path):
    return np.asarray(Image.open(path).convert("RGB"))


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
    args = ap.parse_args()

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
    np.savez_compressed(out,
                        rgb=(data["rgb"] * 255.0).astype(np.uint8),
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
