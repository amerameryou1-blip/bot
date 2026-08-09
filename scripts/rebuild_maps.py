#!/usr/bin/env python3
"""Rebuild sim maps from REAL game screenshots (v2).

v2 fixes found by debugging v1 on real screenshots:
  - UI gray panels (~128,128,128 flat) were classified as MOUNTAIN.
    Now: gray is mountain ONLY if it has local texture (3x3 variance);
    flat gray = UI.
  - Map bbox was the whole screen (leaderboard overlays the map on some
    maps, info panel boxes it in on others). Now: row/col projection of
    map-class pixels finds the map rectangle; dark/light UI is excluded
    from the % stats.
  - Land in the real game is darker/warmer; thresholds re-tuned from
    actual pixel stats (beige + green).
  - Every map prints an ASCII class map for human (and model) verification,
    and % stats are gated against the OCR'd in-game stats (±5%).

Run:  SHOTS_DIR=... python3 scripts/rebuild_maps.py
Output: weights/maps/<slug>.npz (int8: -2 mountain, -1 water, 0 land)
        weights/maps/maps_meta.json, weights/maps/preview_*.png
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SHOTS = Path(os.environ.get("SHOTS_DIR", REPO / ".." / "realdata" / "shots"))
OUT = REPO / "weights" / "maps"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_W = 200
TOL_PCT = 5.0

# screenshot -> (slug, real_w, real_h, expected stats from in-game OCR)
GROUND_TRUTH = {
    "Screenshot 2026-08-07 192345.png": ("island", 510, 510, {}),
    "Screenshot 2026-08-07 192425.png": ("white_arena", 228, 228, {}),
    "Screenshot 2026-08-07 192444.png": ("black_arena", 798, 798, {}),
    "Screenshot 2026-08-07 192505.png": ("mountains", 958, 958, {"land": 65.6, "water": 23.1, "mountain": 11.3}),
    "Screenshot 2026-08-07 192527.png": ("desert", 898, 898, {"land": 82.6, "water": 17.4}),
    "Screenshot 2026-08-07 192545.png": ("swamp", 998, 998, {"land": 63.7, "water": 36.3}),
    "Screenshot 2026-08-07 192606.png": ("white_plains", 998, 998, {"land": 59.9, "water": 40.1}),
    "Screenshot 2026-08-07 192624.png": ("white_plains", 998, 998, {"land": 59.9, "water": 40.1}),
    "Screenshot 2026-08-07 192641.png": ("cliffs", 1022, 1022, {"land": 35.8, "water": 48.7, "mountain": 16.0}),
    "Screenshot 2026-08-07 192657.png": ("pond", 818, 818, {"land": 90.2, "water": 9.8}),
    "Screenshot 2026-08-07 192715.png": ("halo", 1022, 1022, {"land": 41.4, "water": 54.2, "mountain": 4.3}),
    "Screenshot 2026-08-07 192731.png": ("island_kingdom", 1022, 1022, {"land": 12.6, "water": 87.4}),
    "Screenshot 2026-08-07 192747.png": ("mountains2", 938, 938, {"land": 55.7, "mountain": 44.3}),
    "Screenshot 2026-08-07 192807.png": ("europe", 929, 957, {"land": 61.4, "water": 38.6}),
    "Screenshot 2026-08-07 192829.png": ("world", 1754, 998, {"land": 29.6, "water": 70.4}),
    "Screenshot 2026-08-07 192854.png": ("caucasia", 1298, 745, {"land": 72.0, "water": 28.0}),
    "Screenshot 2026-08-07 192907.png": ("africa", 831, 928, {"land": 55.8, "water": 44.2}),
    "Screenshot 2026-08-07 192919.png": ("middle_east", 998, 705, {"land": 70.0, "water": 30.0}),
    "Screenshot 2026-08-07 192935.png": ("scandinavia", 882, 958, {"land": 56.9, "water": 43.1}),
    "Screenshot 2026-08-07 192946.png": ("north_america", 954, 871, {"land": 51.0, "water": 49.0}),
    "Screenshot 2026-08-07 192957.png": ("south_america", 688, 923, {"land": 46.1, "water": 53.9}),
    "Screenshot 2026-08-07 193010.png": ("asia", 938, 828, {"land": 48.3, "water": 51.7}),
    "Screenshot 2026-08-07 193029.png": ("australia", 998, 891, {"land": 45.2, "water": 54.8}),
    "Screenshot 2026-08-07 193043.png": ("world2", 1539, 1078, {"land": 40.1, "water": 59.9}),
    "Screenshot 2026-08-07 193055.png": ("british_isles", 1048, 1390, {"land": 33.2, "water": 66.8}),
    "Screenshot 2026-08-07 193109.png": ("mare_nostrum", 1898, 762, {"land": 52.7, "water": 47.3}),
}


def _box_mean(a: np.ndarray, k: int = 3) -> np.ndarray:
    """Fast kxk box mean via cumsum (values must be float)."""
    h, w = a.shape
    cs = np.cumsum(np.cumsum(a, axis=0), axis=1)
    cs = np.pad(cs, ((k, 0), (k, 0)))  # pad top/left by k so indices never go OOB
    out = np.zeros_like(a)
    for dy in range(k):
        for dx in range(k):
            y0, y1 = dy, h + dy
            x0, x1 = dx, w + dx
            out += (cs[y1, x1] - cs[y0, x1] - cs[y1, x0] + cs[y0, x0]) / (k * k)
    return out


def classify(rgb: np.ndarray, gray_is_mountain: bool = False,
             flat_t: float = 2.0, water_m: int = 10, green_m: int = 5) -> np.ndarray:
    """int8: -2 mountain, -1 water, 0 land, -9 UI/unknown.

    v4 (2026-08-09, tuned by EYES on World1/Mountains1/Europe):
      * page background = PERFECTLY FLAT gray -> variance test -> UI
      * mountain maps: rock+snow = everything not green and not blue
        (snow is WHITE — the old light_land rule was eating it as land)
      * normal maps: gray/beige/white = land (world/caucasia/white_plains)
      * water_m/green_m/flat_t are per-map knobs, auto-calibrated against
        the OCR percentages the game prints on the screenshot."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    H, W = r.shape
    out = np.full((H, W), -9, dtype=np.int8)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    total = r + g + b

    lum = total / 3.0
    m1 = _box_mean(lum, 3)
    m2 = _box_mean(lum * lum, 3)
    flat = (m2 - m1 * m1) < flat_t
    bg = flat   # v4b reverted: mid-gray window ate real gray land

    dark_ui = (total < 90) & ~((g > r + 5) & (g > b + 5)) & ~((b > r + 10) & (b > g + 5))

    water = (b > r + water_m) & (b > g + (water_m // 2 + 2)) & (b >= 40) & ~dark_ui
    green = (g > r + green_m) & (g > b + green_m) & (g > 30)

    out[water] = -1
    if gray_is_mountain:
        # rock + snow = not water, not green, not UI-dark, not page bg
        mountain = ~water & ~green & ~dark_ui & ~bg & (total >= 45)
        out[mountain] = -2
        out[green] = 0
    else:
        chroma_land = (chroma >= 12) & (r >= g) & (g >= b) & (r > 85)
        light_land = (total >= 640) & (chroma < 14)
        gray_land = (chroma < 12) & ~dark_ui & ~bg & (total >= 45) & (mx < 230)
        out[green | chroma_land | light_land | gray_land] = 0
    return out


def calibrate(rgb: np.ndarray, expect: dict, aspect: float):
    """Search (flat_t, water_m, green_m) so extracted stats match the OCR
    percentages printed on the screenshot. Returns (knobs, cls) best fit."""
    gim = expect.get("mountain", 0.0) > 5.0
    best = None
    for flat_t in (0.5, 1.0, 2.0, 4.0):
        for water_m in (5, 10, 16):
            for green_m in (3, 5, 8):
                cls = classify(rgb, gray_is_mountain=gim, flat_t=flat_t,
                               water_m=water_m, green_m=green_m)
                rect = map_rect(cls, aspect=aspect)
                if rect is None:
                    continue
                y0, y1, x0, x1 = _snap(rect, cls.shape, aspect)
                st = stats_of(cls[y0:y1 + 1, x0:x1 + 1])
                err = sum(abs(st.get(k, 0) - expect.get(k, 0))
                          for k in ("land", "water", "mountain") if k in expect)
                if best is None or err < best[0]:
                    best = (err, (flat_t, water_m, green_m), cls)
    return best


def map_rect(cls: np.ndarray, aspect: float | None = None) -> tuple:
    """Find the map rectangle.

    Density search when aspect is known (real map aspect from OCR): the map is
    a rectangle of that aspect, vertically centered, maximizing water+land
    density inside while minimizing it outside. Robust to translucent UI
    panels that overlay the map's edges. Falls back to row/col projection.
    """
    H, W = cls.shape
    is_map = cls != -9   # water OR land OR mountain (v4: mountains count!)
    if aspect is not None and 0.5 < aspect < 3.5:
        # Water exists ONLY on the map -> the map rect must CONTAIN the water
        # bbox. Then snap to the real aspect (vertically centered) and pick
        # the placement with the best water+land density.
        wmask = cls == -1
        if wmask.sum() > 200:
            wy = np.where(wmask.any(axis=1))[0]
            wx = np.where(wmask.any(axis=0))[0]
            wb = (int(wy.min()), int(wy.max()), int(wx.min()), int(wx.max()))
        else:
            wb = None
        best = None
        for frac_h in np.linspace(0.45, 1.0, 32):
            h = int(H * frac_h)
            w = int(h * aspect)
            if w <= 10:
                continue
            # wide real maps (world/mare_nostrum/...) fill the ENTIRE screen
            # width — clamp instead of skipping (v3 fix, 2026-08-09)
            w = min(w, W - 1)
            y0 = (H - h) // 2
            y1 = min(H, y0 + h)
            if wb is not None:
                if h < wb[1] - wb[0] + 1:
                    continue
                lo = max(0, wb[3] - w + 1)
                hi = min(W - w, wb[2])
                if lo > hi:
                    continue
                xs = np.linspace(lo, hi, max(2, int((hi - lo) / max(1, W // 30)) + 1))
            else:
                xs = np.linspace(0, W - w, 11)
            for x0 in xs:
                x0 = int(x0)
                x1 = min(W, x0 + w)
                inside = float(is_map[y0:y1, x0:x1].mean())
                left = float(is_map[y0:y1, :x0].mean()) if x0 > 0 else 0.0
                right = float(is_map[y0:y1, x1:].mean()) if x1 < W else 0.0
                score = inside - 0.5 * (left + right)
                if best is None or score > best[0]:
                    best = (score, (y0, y1, x0, x1))
        if best is not None and best[0] > 0.15:
            return best[1]
    # fallback: projection
    rowsum = is_map.sum(axis=1).astype(np.float32)
    colsum = is_map.sum(axis=0).astype(np.float32)
    rmax = rowsum.max() if rowsum.max() > 0 else 1
    cmax = colsum.max() if colsum.max() > 0 else 1
    rows = np.where(rowsum > 0.30 * rmax)[0]
    cols = np.where(colsum > 0.30 * cmax)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _snap(rect, shape, real_aspect):
    """Trim a rect to the real map aspect ratio (centered)."""
    y0, y1, x0, x1 = rect
    ch = y1 - y0 + 1
    cw = x1 - x0 + 1
    cur_aspect = cw / ch
    if cur_aspect > real_aspect:          # too wide -> trim columns
        cw_t = int(round(ch * real_aspect))
        cx0 = x0 + (cw - cw_t) // 2
        x0, x1 = max(0, cx0), min(shape[1] - 1, cx0 + cw_t)
    else:                                  # too tall -> trim rows
        ch_t = int(round(cw / real_aspect))
        cy0 = y0 + (ch - ch_t) // 2
        y0, y1 = max(0, cy0), min(shape[0] - 1, cy0 + ch_t)
    return y0, y1, x0, x1


def stats_of(cls: np.ndarray) -> dict:
    valid = cls >= -2
    n = int(valid.sum())
    if n == 0:
        return {"land": 0.0, "water": 0.0, "mountain": 0.0, "ui": 100.0}
    land = float((cls == 0).sum() / n * 100)
    water = float((cls == -1).sum() / n * 100)
    mountain = float((cls == -2).sum() / n * 100)
    ui = float((cls == -9).sum() / cls.size * 100)
    return {"land": land, "water": water, "mountain": mountain, "ui": ui}


def block_majority(cls: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Downsample to (th, tw) by MAJORITY class per block.
    v4: UI/background (-9) becomes MOUNTAIN (-2, impassable) — the page
    background around real maps must block movement, not be free land."""
    H, W = cls.shape
    out = np.zeros((th, tw), dtype=np.int8)
    for yy in range(th):
        y0 = yy * H // th; y1 = max(yy + 1, (yy + 1) * H // th)
        for xx in range(tw):
            x0 = xx * W // tw; x1 = max(xx + 1, (xx + 1) * W // tw)
            block = cls[y0:y1, x0:x1]
            vals, counts = np.unique(block, return_counts=True)
            best = vals[int(np.argmax(counts))]
            out[yy, xx] = best if best != -9 else -2
    return out


def ascii_map(cls: np.ndarray, w: int = 100) -> str:
    h = max(8, int(cls.shape[0] / cls.shape[1] * w * 0.5))
    shifted = (cls + 9).astype(np.uint8)  # 0=UI 1=water 2=land 7=mountain
    img = Image.fromarray(shifted).resize((w, h), Image.NEAREST)
    a = np.asarray(img, dtype=np.int16) - 9
    ch = {-2: 'M', -1: 'w', 0: 'L', -9: '.'}
    return '\n'.join(''.join(ch.get(int(c), '?') for c in row) for row in a)


def main() -> int:
    failures = 0
    meta = {}
    for fname, (slug, rw, rh, expect) in sorted(GROUND_TRUTH.items()):
        src = SHOTS / fname
        if not src.exists():
            print(f"[{slug:16s}] SKIP: missing {src.name}")
            continue
        rgb = np.asarray(Image.open(src).convert('RGB'))
        aspect = rw / rh
        if expect:
            fit = calibrate(rgb, expect, aspect)
            if fit is None:
                print(f"[{slug:16s}] FAIL: no map rect")
                failures += 1
                continue
            err, knobs, cls = fit
        else:
            knobs = None
            cls = classify(rgb, gray_is_mountain=expect.get("mountain", 0.0) > 5.0)
        rect = map_rect(cls, aspect=aspect)
        if rect is None:
            print(f"[{slug:16s}] FAIL: no map rect")
            failures += 1
            continue
        y0, y1, x0, x1 = _snap(rect, cls.shape, aspect)
        crop = cls[y0:y1 + 1, x0:x1 + 1]
        ch, cw = crop.shape

        tw = TARGET_W
        th = max(8, int(round(tw * ch / cw)))
        small = block_majority(crop, tw, th)   # UI -9 inside the map -> land 0

        # stats gate on FULL-RES crop (not the downsampled grid)
        stats = stats_of(crop)
        print(f"[{slug:16s}] sim {tw}x{th}  land={stats['land']:.1f} water={stats['water']:.1f} "
              f"mt={stats['mountain']:.1f} ui={stats['ui']:.1f}   (real {rw}x{rh} {expect})")

        ok = True
        for key, tol in (("land", TOL_PCT), ("water", TOL_PCT), ("mountain", TOL_PCT)):
            if expect.get(key) is not None and abs(stats[key] - expect[key]) > tol:
                ok = False
        print(f"    {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures += 1

        np.savez_compressed(OUT / f"{slug}.npz", world=small)
        meta[slug] = {
            "file": fname, "real_w": rw, "real_h": rh,
            "sim_w": small.shape[1], "sim_h": small.shape[0],
            "stats_ocr": expect, "stats_extracted": stats,
            "pass": ok,
            "ascii": ascii_map(small, 100),
        }

        pal = np.zeros((*small.shape, 3), dtype=np.uint8)
        pal[small == -1] = (20, 90, 150)
        pal[small == -2] = (120, 110, 100)
        pal[small == 0] = (214, 205, 160)
        pal[small == -9] = (30, 30, 30)
        Image.fromarray(pal).save(OUT / f"preview_{slug}.png")

    with open(OUT / "maps_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n{'ALL MAPS PASS' if failures == 0 else f'{failures} MAP(S) FAILED — DO NOT TRAIN ON FAILED MAPS'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
