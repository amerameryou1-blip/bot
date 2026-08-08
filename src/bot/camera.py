"""Camera control for territorial.io — the zoom-out fix (v2, empirical).

How the game camera works (verified in the 657KB inline game script):
  * The game renders on `canvasA`; wheel zoom is a DOM listener:
        canvasA.addEventListener('wheel', fb)
        fb(eL) -> eZ.fb(clientX, clientY, deltaY)
               -> fi = 500/(500+deltaY), clamped to [0.5, 2] per tick
               -> du.zoom(fi, ...) + fk(fi, ...)   (the real camera transform)
  * `du` is NOT global (private `new hb()` in the IIFE) — but a synthetic
    WheelEvent on canvasA runs the game's OWN wheel path. No isTrusted check.
  * The zoom is centered on the cursor (clientX/clientY) — so dispatching the
    wheel event AT OUR BLOB'S screen position zooms toward our territory.
  * Camera reset (iJ) only happens on resize/menu. No per-frame auto-follow.
  * Wheel input is ignored while a zoom animation runs (eZ.eT()) — space ticks.

Empirics (match #1, Island): at match start the game auto-fits the camera to
a wide view: our blob was ~0.46% of the screen (target 1-6%) with enemy
territories visible. Zooming OUT (old v1) shrank the blob further — wrong
direction. v2 searches BOTH directions, zooming toward our blob's position.
"""
from __future__ import annotations

import os
import time

import numpy as np

from bot.calibration import _ocr_words, blob, find_name_box

_WHEEL_OUT = 800      # deltaY>0 -> fi=0.5 -> 2x zoom OUT
_WHEEL_IN = -800      # deltaY<0 -> fi=2.0 -> 2x zoom IN
_TICK_DELAY_S = 0.7
TARGET_MIN, TARGET_MAX = 0.01, 0.06   # self blob fraction of screen (user: 1-5%)
# ZOOM_LEVEL env for data variety: "auto" = fit-to-target window,
# 0 = leave the game's auto-fit view, 1/2/... = force N zoom-in ticks.
ZOOM_LEVEL = os.environ.get("ZOOM_LEVEL", "auto").strip().lower()


def _dispatch_wheel(page, x: int, y: int, delta: int) -> bool:
    """Run the game's own wheel handler at (x,y) via a synthetic event."""
    return bool(page.evaluate(
        """(o) => {
            const c = document.getElementById('canvasA');
            if (!c) return false;
            c.dispatchEvent(new WheelEvent('wheel', {
                deltaY: o.d, clientX: o.x, clientY: o.y,
                bubbles: true, cancelable: true
            }));
            return true;
        }""", {"x": x, "y": y, "d": delta}
    ))


# Visible on-canvas zoom buttons (1280x800 viewport) — trusted-click path.
# Verified BY EYE in recorded frames (2026-08-08): round +/- buttons on the
# right edge, vertically centred. page.mouse.click is trusted by the canvas
# (unlike page.mouse.wheel, which the game ignores).
ZOOM_BTN_IN = (1240, 363)
ZOOM_BTN_OUT = (1240, 443)


def _click_zoom(page, direction: str) -> bool:
    """Click the game's visible zoom button (trusted mouse event)."""
    x, y = ZOOM_BTN_IN if direction == "in" else ZOOM_BTN_OUT
    try:
        page.mouse.click(x, y)
        return True
    except Exception:
        return False


def _lighten(c, t=0.55):
    """The game renders OUR territory as a lightened tint of the swatch
    color (verified by eye 2026-08-08: swatch [48,180,24] vs on-map mint
    ~[154,205,166]). A swatch-only mask sees just the border ring and
    reports ~1.8% while the view is actually zoomed hard into our spawn."""
    return [int(v + (255 - v) * t) for v in c]


def _self_blob(img: np.ndarray, self_rgb, tol: int = 24):
    """Union of the sampled shade + its lighter/darker family (the game
    renders own-territory interiors/edges in different shades; the exact
    split is inconsistent — cover the family instead of theorizing)."""
    shades = [list(self_rgb), _lighten(self_rgb), [int(v * 0.65) for v in self_rgb]]
    masks, best = [], None
    for c in shades:
        b = blob(img, c, tol=tol, min_area=10)
        if b:
            masks.append(b["mask"])
            if best is None or b["area"] > best["area"]:
                best = b
    if not masks:
        return None
    m = masks[0]
    for mm in masks[1:]:
        m = m | mm
    c = np.argwhere(m)
    return {"area": len(c), "cy": float(c[:, 0].mean()),
            "cx": float(c[:, 1].mean()), "mask": m}


def self_blob_frac(img: np.ndarray, self_rgb, tol: int = 24) -> float:
    """Our territory blob as a fraction of the screen (0..1)."""
    b = _self_blob(img, self_rgb, tol)
    if not b:
        return 0.0
    H, W = img.shape[:2]
    return b["area"] / (H * W)


def _distinct_enemy_clusters(img: np.ndarray, self_rgb,
                             tol: int = 20, cluster_px: int = 60,
                             color_dist: int = 90) -> list:
    """Distinct enemy territory clusters (dedupes antialiased color variants).

    Two blobs are the SAME enemy if their centers are within `cluster_px`
    OR their colors are within `color_dist` (perceptual-ish RGB distance).

    False-positive filters (empirically needed):
      * colors too close to OUR color (antialiased self-territory shades)
      * blob centers inside our own territory blob
      * blob centers in UI zones (leaderboard top-left, bottom bar, top strip)
      * colors close to the frame's dominant background (water/terrain)
    """
    from bot.calibration import edges_touched, saturated_colors

    H, W = img.shape[:2]
    self_b = blob(img, list(self_rgb), tol=24, min_area=10)
    self_rect = None
    if self_b:
        ys, xs = np.argwhere(self_b["mask"]).T
        self_rect = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    # the largest blob in the frame = background (water/terrain), only treated
    # as background if it dominates (>25%). Enemy territories are smaller.
    bg_color = None
    for c in saturated_colors(img, max_colors=60):
        b = blob(img, c, tol=20, min_area=100)
        if b and b["area"] > 0.25 * H * W:
            bg_color = c
            break

    def in_ui(x, y):
        if x < 470 and y < 330:      # leaderboard / top-left panel
            return True
        if y > H - 55 and 380 < x < 900:  # bottom centre troop bar
            return True
        if y < 40:                   # top banner / player count
            return True
        return False

    blobs = []
    for c in saturated_colors(img, max_colors=60):
        if tuple(c) == tuple(self_rgb):
            continue
        if sum(abs(a - b) for a, b in zip(c, self_rgb)) < 40:
            continue  # too close to our color -> antialiased self territory
        if max(c) < 130:
            continue  # too dim -> dark terrain/water (enemies are vivid)
        if bg_color is not None and sum(abs(a - b) for a, b in zip(c, bg_color)) < 60:
            continue  # the dominant background colour family
        b = blob(img, c, tol=tol, min_area=15)
        if not b or b["area"] > 0.4 * H * W or edges_touched(b["mask"]) >= 3:
            continue
        if in_ui(b["cx"], b["cy"]):
            continue
        if self_rect and (self_rect[0] <= b["cx"] <= self_rect[2]
                          and self_rect[1] <= b["cy"] <= self_rect[3]):
            continue  # inside our territory
        blobs.append((b["cx"], b["cy"], tuple(int(v) for v in c), b["area"]))

    clusters = []
    for bx, by, bc, ba in blobs:
        placed = False
        for cl in clusters:
            cx, cy, cc = cl[0], cl[1], cl[2]
            d_xy = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            d_c = sum(abs(a - b) for a, b in zip(bc, cc))
            if d_xy < cluster_px or d_c < color_dist:
                # merge (keep the bigger blob's center)
                if ba > cl[3]:
                    cl[0], cl[1], cl[2], cl[3] = bx, by, bc, ba
                placed = True
                break
        if not placed:
            clusters.append([bx, by, bc, ba])
    return clusters


def verify_view(img: np.ndarray, self_rgb, tol: int = 24) -> dict:
    """Numeric view check — the success criterion, as numbers.

    PASS: our blob is TARGET_MIN..TARGET_MAX of the screen AND >=2 distinct
    enemy territory clusters are visible.
    """
    H, W = img.shape[:2]
    frac = self_blob_frac(img, self_rgb, tol)
    enemies = _distinct_enemy_clusters(img, self_rgb)
    ok = (TARGET_MIN <= frac <= TARGET_MAX) and len(enemies) >= 2
    return {
        "pass": ok,
        "self_frac": round(frac * 100, 2),
        "self_frac_ok": TARGET_MIN <= frac <= TARGET_MAX,
        "enemies_visible": len(enemies),
        "enemy_clusters": [{"center": [e[0], e[1]], "color": list(e[2])} for e in enemies],
    }


def click_highlighted_row(page) -> bool:
    """Our leaderboard row is the ONLY one with a vivid highlight band
    (green in every observed match, whatever our territory color is).
    Click it — zero OCR, zero name matching."""
    try:
        img = np.array(page.screenshot())
        reg = img[60:310, 10:270].astype(int)
        r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
        m = (g > 70) & (g - r > 30) & (g - b > 30)
        rows = m.any(axis=1)
        if rows.sum() < 6:
            return False
        # contiguous runs; a real row band is 10-40 px tall
        ys = np.flatnonzero(rows)
        runs, start = [], ys[0]
        for a, b2 in zip(ys, ys[1:]):
            if b2 - a > 2:
                runs.append((start, a))
                start = b2
        runs.append((start, ys[-1]))
        runs = [(a, b2) for a, b2 in runs if 8 <= b2 - a <= 45]
        if not runs:
            return False
        a, b2 = max(runs, key=lambda t: t[1] - t[0])
        cy = 60 + int((a + b2) / 2)
        page.mouse.click(140, cy)
        time.sleep(0.6)
        return True
    except Exception:
        return False


def recenter_via_leaderboard(page, bot_name: str, retries: int = 2) -> bool:
    """Click our own row in the leaderboard to recenter the camera on us.
    Name-OCR first (with retries), then the OCR-free highlighted-row click."""
    for _ in range(retries):
        try:
            img = np.array(page.screenshot())
            words = _ocr_words(img)
            if words:
                box, ratio = find_name_box(words, bot_name)
                if box is not None:
                    x, y = box[1] + box[3] // 2, box[2] + box[4] // 2
                    page.mouse.click(x, y)
                    time.sleep(0.6)
                    return True
        except Exception:
            pass
    return click_highlighted_row(page)


def fix_camera(page, grab, self_rgb, bot_name: str | None = None,
               max_ticks: int = 10, log=print) -> dict:
    """Recentre (best-effort) then search the zoom level so our blob is
    TARGET_MIN..TARGET_MAX of the screen with >=2 enemies visible.

    Zoom is dispatched AT our blob's screen position, so the camera zooms
    toward us even if we are off-centre.
    """
    # 0) baseline
    img = grab()
    base = self_blob_frac(img, self_rgb)
    log(f"[camera] baseline self blob = {base * 100:.2f}% of screen")

    # 1) recentre on our leaderboard row (user's trick; best-effort)
    if bot_name and recenter_via_leaderboard(page, bot_name):
        log("[camera] recentered via leaderboard name")
        time.sleep(0.8)
        img = grab()

    # 2) zoom search
    result = None
    ticks = 0
    frac = base
    if ZOOM_LEVEL in ("auto", ""):
        # Stepped search: 1.33x steps (deltaY=-400 -> fi=1.33). Stop at the
        # FIRST zoom where self >=1% AND >=2 enemies; if self >=6% first,
        # step back out. Converges to the closest usable view.
        _STEP_IN = -166   # fi=(500+166)/500 = 1.33x zoom in
        _STEP_OUT = 166   # fi=500/666 = 0.75x zoom out
        while ticks < max_ticks:
            result = verify_view(img, self_rgb)
            if result["pass"]:
                break
            if result["self_frac"] >= TARGET_MAX * 100:
                delta = _STEP_OUT
            elif result["self_frac"] < TARGET_MIN * 100:
                delta = _STEP_IN
            else:
                # self in window but <2 enemies -> widen slightly to see more
                delta = _STEP_OUT
            b = _self_blob(img, self_rgb)
            tx, ty = (int(b["cx"]), int(b["cy"])) if b else (640, 400)
            if not _dispatch_wheel(page, tx, ty, delta):
                log("[camera] FAIL: canvasA not found — cannot zoom")
                break
            ticks += 1
            time.sleep(_TICK_DELAY_S)
            img = grab()
            new_frac = self_blob_frac(img, self_rgb)
            if abs(new_frac - frac) < 1e-4:
                log(f"[camera] wheel stalled at {frac * 100:.2f}% — trusted-button fallback")
                _click_zoom(page, "out")
                time.sleep(_TICK_DELAY_S)
                _click_zoom(page, "out")
                time.sleep(_TICK_DELAY_S)
                if bot_name:
                    recenter_via_leaderboard(page, bot_name)
                    time.sleep(0.6)
                img = grab()
                new_frac = self_blob_frac(img, self_rgb)
                if abs(new_frac - frac) < 1e-4:
                    log("[camera] still stalled — stopping search")
                    break
            frac = new_frac
            log(f"[camera] step {ticks} ({'IN' if delta < 0 else 'OUT'}): "
                f"self={frac * 100:.2f}%")
    else:
        # forced N ticks IN (data variety: ZOOM_LEVEL=1/2/...)
        try:
            force = int(ZOOM_LEVEL)
        except ValueError:
            force = 1
        for i in range(force):
            b = _self_blob(img, self_rgb)
            tx, ty = (int(b["cx"]), int(b["cy"])) if b else (640, 400)
            if not _dispatch_wheel(page, tx, ty, _WHEEL_IN):
                break
            ticks += 1
            time.sleep(_TICK_DELAY_S)
            img = grab()
            # retry once if the tick did not register (game animation may
            # block wheel input right after spawn)
            if self_blob_frac(img, self_rgb) <= frac * 1.1:
                time.sleep(1.5)
                b = _self_blob(img, self_rgb)
                tx, ty = (int(b["cx"]), int(b["cy"])) if b else (640, 400)
                _dispatch_wheel(page, tx, ty, _WHEEL_IN)
                time.sleep(_TICK_DELAY_S)
                img = grab()
        frac = self_blob_frac(img, self_rgb)
        log(f"[camera] forced ZOOM_LEVEL={force}: self={frac * 100:.2f}%")
        # clamp overshoot / under-shoot: if forced ticks didn't register or
        # overshot, fall back to the stepped auto search
        if frac > TARGET_MAX or frac < TARGET_MIN * 0.5:
            log("[camera] forced zoom out of window — running stepped auto search")
            _STEP_IN, _STEP_OUT = -166, 166
            for _ in range(8):
                result = verify_view(img, self_rgb)
                if result["pass"]:
                    break
                if result["self_frac"] >= TARGET_MAX * 100:
                    delta = _STEP_OUT
                elif result["self_frac"] < TARGET_MIN * 100:
                    delta = _STEP_IN
                else:
                    delta = _STEP_OUT
                b = _self_blob(img, self_rgb)
                tx, ty = (int(b["cx"]), int(b["cy"])) if b else (640, 400)
                if not _dispatch_wheel(page, tx, ty, delta):
                    break
                time.sleep(_TICK_DELAY_S)
                img = grab()
                new_frac = self_blob_frac(img, self_rgb)
                if abs(new_frac - frac) < 1e-4:
                    break
                frac = new_frac
                log(f"[camera] clamp step ({'IN' if delta < 0 else 'OUT'}): "
                    f"self={frac * 100:.2f}%")
            log(f"[camera] after clamp: self={frac * 100:.2f}%")

    # v3: trusted-button escape from EXTREME zoom (we're inside our own
    # territory, giant name text on screen — wheel search can stall there).
    img = grab()
    frac = self_blob_frac(img, self_rgb)
    btn = 0
    while frac > TARGET_MAX and btn < 10:
        _click_zoom(page, "out")
        btn += 1
        time.sleep(_TICK_DELAY_S)
        img = grab()
        frac = self_blob_frac(img, self_rgb)
        log(f"[camera] button OUT {btn}: self={frac * 100:.2f}%")
    if btn and bot_name:
        recenter_via_leaderboard(page, bot_name)
        time.sleep(0.6)
        img = grab()

    if result is None:
        result = verify_view(img, self_rgb)
    log(f"[camera] FINAL: pass={result['pass']} self={result['self_frac']}% "
        f"enemies={result['enemies_visible']}")
    return result
