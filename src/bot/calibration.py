"""Per-match color calibration — reads MY territory color from the leaderboard.

territorial.io assigns a NEW color every match, so a hardcoded color goes stale
after one game. The robust fix: every match, the leaderboard (top-left) shows
my name with a color swatch right before it. We OCR the leaderboard, find my
name, read the swatch color, and validate it's a real territory blob in the
frame. Falls back to center-lock detection if OCR is unavailable.

Pure-numpy helpers are unit-testable without tesseract; the OCR path degrades
gracefully if tesseract isn't installed.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

try:
    import pytesseract
    from PIL import Image, ImageOps

    _HAS_OCR = True
except Exception:  # pragma: no cover - environment without OCR deps
    _HAS_OCR = False

from .config import Palette, PlayerColor

# Leaderboard region (top-left) OCR'd for the name row.
LB_REGION = (0, 0, 460, 320)
# How far left of the name the swatch can be.
SWATCH_MAX_DX = 34
SWATCH_MIN_DX = 2


def saturated_colors(img: np.ndarray, max_colors: int = 24, sat_min: int = 55,
                     bright_min: int = 110) -> list[list[int]]:
    """Quantized saturated colors present in the frame (player territory colors)."""
    px = img[::2, ::2].reshape(-1, 3).astype(int)
    mx = px.max(axis=1) - px.min(axis=1)
    mask = (mx > sat_min) & (px.max(axis=1) > bright_min)
    if not mask.any():
        return []
    q = px[mask] // 24 * 24
    return [list(c) for c, _ in Counter(map(tuple, q)).most_common(max_colors)]


def blob(img: np.ndarray, color, tol: int = 48, min_area: int = 150):
    """Largest mask of pixels within `tol` of `color`; None if too small."""
    m = np.all(np.abs(img.astype(int) - np.array(color)) < tol, axis=2)
    c = np.argwhere(m)
    if len(c) < min_area:
        return None
    return {"area": len(c), "cy": float(c[:, 0].mean()), "cx": float(c[:, 1].mean()), "mask": m}


def edges_touched(mask: np.ndarray) -> int:
    """How many of the 4 screen edges the mask touches (backgrounds touch ≥3)."""
    h, w = mask.shape
    n = 0
    if mask[0, :].any():
        n += 1
    if mask[-1, :].any():
        n += 1
    if mask[:, 0].any():
        n += 1
    if mask[:, -1].any():
        n += 1
    return n


def swatch_from_strip(strip_rgb: np.ndarray, min_frac: float = 0.03) -> list[int] | None:
    """Pick the most-saturated color in a strip (the swatch, not the panel bg).

    The leaderboard strip between the rank and the name contains the swatch
    (small, highly saturated) on the panel background (dark, low saturation).
    """
    if len(strip_rgb) < 30:
        return None
    q = strip_rgb // 12 * 12
    counts = Counter(map(tuple, q))
    total = len(strip_rgb)
    cands = []
    for c, n in counts.most_common(12):
        if n / total < min_frac:
            continue
        mx = max(c) - min(c)
        if mx < 40 or max(c) < 90:
            continue
        cands.append((mx, n, list(c)))
    if not cands:
        return None
    cands.sort(key=lambda x: (-x[0], -x[1]))
    return cands[0][2]


def validate_territory_color(img: np.ndarray, color, tol: int = 48,
                             min_area: int = 15, min_outside_lb: int = 8) -> bool:
    """A real territory color forms a blob outside the leaderboard region.

    Relaxed for the very start of a match: your territory begins at 12 pixels,
    so a small blob is correct. The swatch itself lives inside the leaderboard
    region (top-left), so we require the color to also appear OUTSIDE it.
    """
    H, W = img.shape[:2]
    b = blob(img, color, tol=tol, min_area=min_area)
    if b is None:
        return False
    frac = b["area"] / (H * W)
    if frac > 0.4 or edges_touched(b["mask"]) >= 3:
        return False
    m = b["mask"]
    outside_lb = m.copy()
    outside_lb[:340, :480] = False
    if outside_lb.sum() < min_outside_lb:
        return False
    return True


def _ocr_words(img: np.ndarray):
    """OCR the leaderboard region; yield (word, x, y, w, h) in original coords."""
    if not _HAS_OCR:
        return []
    pil = Image.fromarray(img)
    x0, y0, x1, y1 = LB_REGION
    region = pil.crop((x0, y0, x1, y1))
    scale = 3
    region = region.resize((region.width * scale, region.height * scale), Image.LANCZOS)
    data = pytesseract.image_to_data(ImageOps.grayscale(region), config="--psm 6",
                                     output_type=pytesseract.Output.DICT)
    out = []
    for i in range(len(data["text"])):
        t = data["text"][i].strip()
        if not t:
            continue
        out.append((t,
                    data["left"][i] // scale, data["top"][i] // scale,
                    data["width"][i] // scale, data["height"][i] // scale))
    return out


def find_name_box(words, bot_name: str, min_ratio: float = 0.45):
    """Best OCR word matching the bot name (fuzzy), or None."""
    import difflib

    best, best_ratio = None, 0.0
    for w in words:
        t = w[0]
        ratio = difflib.SequenceMatcher(None, t.lower(), bot_name.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = w
    if best is None or best_ratio < min_ratio:
        return None, best_ratio
    return best, best_ratio


def calibrate_from_leaderboard(img: np.ndarray, bot_name: str, tol: int = 48):
    """Read my territory color from the leaderboard swatch.

    Returns (Palette | None, reason_string).
    """
    words = _ocr_words(img)
    if not words:
        return None, "ocr unavailable/no words"
    name_box, ratio = find_name_box(words, bot_name)
    if name_box is None:
        return None, f"name not found (best ratio {ratio:.2f})"
    _, nx, ny, _nw, nh = name_box

    sx0 = max(0, nx - SWATCH_MAX_DX)
    sx1 = max(0, nx - SWATCH_MIN_DX)
    strip = img[ny:ny + max(nh, 4), sx0:sx1].reshape(-1, 3).astype(int)
    swatch = swatch_from_strip(strip)
    if swatch is None:
        return None, "no saturated swatch in strip"

    if not validate_territory_color(img, swatch, tol=tol):
        return None, f"swatch {swatch} not a valid territory blob"

    H, W = img.shape[:2]
    enemies = []
    for c in saturated_colors(img, max_colors=34):
        if tuple(c) == tuple(swatch):
            continue
        b = blob(img, c, min_area=15)
        if b and b["area"] < 0.4 * H * W and edges_touched(b["mask"]) < 3:
            enemies.append(PlayerColor(f"e{len(enemies)}", *c))
        if len(enemies) >= 10:
            break
    palette = Palette(self_color=PlayerColor("me", *swatch), enemy_colors=enemies,
                      tolerance=tol, downscale=2)
    return palette, f"leaderboard swatch={swatch} area={blob(img, swatch, tol)['area']}"


def center_lock_calibrate(page, grab, timeout_s: float = 30.0):
    """Fallback: wait for a saturated blob locked to the screen center.

    Works when the match camera follows my territory (my color occupies the
    center and stays there while the world moves).
    """
    H, W = 800, 1280
    cy, cx = H // 2, W // 2
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        img = grab()
        center = img[cy, cx].astype(int).tolist()
        if max(center) < 90 or (max(center) - min(center)) < 40:
            time.sleep(0.5)
            continue
        cols = saturated_colors(img)
        if not cols:
            time.sleep(0.5)
            continue
        best = min(cols, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, center)))
        b = blob(img, best)
        if not b or b["area"] > 0.38 * H * W or edges_touched(b["mask"]) >= 3:
            time.sleep(0.5)
            continue
        # verify it stays centered while the world moves
        page.keyboard.down("ArrowRight")
        time.sleep(0.4)
        page.keyboard.up("ArrowRight")
        time.sleep(0.2)
        img2 = grab()
        b2 = blob(img2, best)
        if not b2 or not b2["mask"][cy, cx]:
            continue
        disp = ((b["cy"] - b2["cy"]) ** 2 + (b["cx"] - b2["cx"]) ** 2) ** 0.5
        if disp > 25:
            continue
        enemies = []
        for c in saturated_colors(img2, max_colors=34):
            if tuple(c) == tuple(best):
                continue
            bb = blob(img2, c)
            if bb and 150 < bb["area"] < 0.38 * H * W and edges_touched(bb["mask"]) < 3:
                enemies.append(PlayerColor(f"e{len(enemies)}", *c))
            if len(enemies) >= 10:
                break
        return Palette(self_color=PlayerColor("me", *best), enemy_colors=enemies,
                       tolerance=48.0, downscale=2)
    return None
