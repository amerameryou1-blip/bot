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


def dominant_colors(img: np.ndarray, n: int = 6, min_frac: float = 0.03) -> list[list[int]]:
    """The most common colors in the frame that cover a meaningful fraction.

    Terrain/UI backgrounds dominate (>3% of the frame); player territories
    (12 px at match start, ~0.01%) never do. Used to reject false swatch reads.
    """
    px = img[::3, ::3].reshape(-1, 3).astype(int)
    total = len(px)
    q = (px // 24 * 24)
    return [list(c) for c, cnt in Counter(map(tuple, q)).most_common(n) if cnt / total >= min_frac]


def swatch_from_strip(strip_rgb: np.ndarray, min_frac: float = 0.03,
                      bright_min: int = 140) -> list[int] | None:
    """Pick the most-saturated color in a strip (the swatch, not the panel bg).

    The leaderboard strip between the rank and the name contains the swatch
    (small, highly saturated, BRIGHT) on the panel background (dark, dull).
    Player colors are vivid, so we require a bright channel — this rejects
    dark terrain-ish colors like the map's [0,96,12].
    """
    if len(strip_rgb) < 30:
        return None
    q = strip_rgb // 12 * 12
    counts = Counter(map(tuple, q))
    total = len(strip_rgb)
    cands = []
    for c, n in counts.most_common(16):
        if n / total < min_frac:
            continue
        mx = max(c) - min(c)
        if mx < 60 or max(c) < bright_min:
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

    A color that is among the frame's DOMINANT colors is terrain/UI, not a
    territory — rejected regardless of blob size.
    """
    H, W = img.shape[:2]
    # reject terrain/UI: a territory color never dominates the whole frame
    for dc in dominant_colors(img, n=6):
        if np.all(np.abs(np.array(color) - np.array(dc)) <= 24):
            return False
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

    Strategy: OCR every word row in the leaderboard region, read the color
    strip left of each row, and VALIDATE each candidate. The correct row is
    the one whose swatch color is a real (bright, non-terrain) territory blob
    on the map. We prefer the row whose name matches `bot_name` best, but we
    don't depend on OCR getting the name exactly right — we scan all rows and
    pick the first that validates.

    Returns (Palette | None, reason_string).
    """
    words = _ocr_words(img)
    if not words:
        return None, "ocr unavailable/no words"

    # Group OCR words into rows (same y band).
    rows: list[dict] = []
    for w in words:
        t, x, y, ww, hh = w
        placed = False
        for r in rows:
            if abs(r["y"] - y) < 14:
                r["words"].append(w)
                r["y"] = min(r["y"], y)
                placed = True
                break
        if not placed:
            rows.append({"y": y, "words": [w]})
    if not rows:
        return None, "no rows"

    import difflib

    # Score each row: does it contain a word close to my name?
    def row_name_score(row) -> float:
        best = 0.0
        for w in row["words"]:
            ratio = difflib.SequenceMatcher(None, w[0].lower(), bot_name.lower()).ratio()
            best = max(best, ratio)
        return best

    for row in sorted(rows, key=row_name_score, reverse=True):
        score = row_name_score(row)
        name_x = min(w[1] for w in row["words"])
        ny = row["y"]
        nh = max((w[4] for w in row["words"]), default=12)
        # Scan the whole row band (rank + swatch + name) for the BRIGHTEST
        # saturated color — the swatch is small but vivid; the row background
        # is dark and gets filtered out by the brightness floor.
        row_band = img[ny:ny + max(nh + 6, 12), max(0, name_x - 60):name_x - 2].reshape(-1, 3).astype(int)
        swatch = swatch_from_strip(row_band)
        if swatch is None:
            continue
        if not validate_territory_color(img, swatch, tol=tol):
            continue

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
        return palette, f"leaderboard row(name_match={score:.2f}) swatch={swatch}"

    # No row validated — diagnostics for the closest name row
    best_row = max(rows, key=row_name_score)
    name_x = min(w[1] for w in best_row["words"])
    ny = best_row["y"]
    nh = max((w[4] for w in best_row["words"]), default=12)
    band = img[ny:ny + max(nh + 6, 12), max(0, name_x - 60):name_x - 2].reshape(-1, 3).astype(int)
    swatch = swatch_from_strip(band)
    H, W = img.shape[:2]
    detail = ""
    if swatch is not None:
        b = blob(img, swatch, tol=tol, min_area=3)
        if b:
            detail = f" (area={b['area']}, frac={b['area']/(H*W):.4f}, edges={edges_touched(b['mask'])})"
    hint = ""
    if swatch is None:
        hint = " (no BRIGHT saturated color in your leaderboard row — your color may be dark/terrain-like; pick a vivid color in the editor and set MANUAL_COLOR)"
    return None, f"no row validated (closest name score={row_name_score(best_row):.2f}, swatch={swatch}{detail}){hint}"


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


def ocr_map_info(img: np.ndarray):
    """OCR the 'Map: X / Dimension: WxH' text on the start screen (bottom-center).

    Returns {"name": str, "w": int, "h": int} or None. Best-effort — used only
    to tag recordings with the map identity.
    """
    if not _HAS_OCR:
        return None
    try:
        from PIL import ImageOps
        import pytesseract
        pil = Image.fromarray(img)
        H, W = pil.size
        region = pil.crop((int(W * 0.30), int(H * 0.72), int(W * 0.75), int(H * 0.96)))
        region = region.resize((region.width * 3, region.height * 3), Image.LANCZOS)
        data = pytesseract.image_to_data(ImageOps.grayscale(region), config="--psm 6",
                                         output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data["text"])):
            t = data["text"][i].strip()
            if t:
                words.append(t)
        text = " ".join(words)
        out = {"raw": text}
        import re
        m = re.search(r"Map:\s*([A-Za-z0-9 _\-]+)", text)
        if m:
            out["name"] = m.group(1).strip()
        m = re.search(r"Dimension:\s*(\d+)\s*x\s*(\d+)", text)
        if m:
            out["w"], out["h"] = int(m.group(1)), int(m.group(2))
        return out if ("name" in out or "w" in out) else None
    except Exception:
        return None
