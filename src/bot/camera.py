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


def _self_blob(img: np.ndarray, self_rgb, tol: int = 24):
    return blob(img, list(self_rgb), tol=tol, min_area=10)


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


def recenter_via_leaderboard(page, bot_name: str) -> bool:
    """Click our own row in the leaderboard to recenter the camera on us."""
    try:
        img = np.array(page.screenshot())
        words = _ocr_words(img)
        if not words:
            return False
        box, ratio = find_name_box(words, bot_name)
        if box is None:
            return False
        x, y = box[1] + box[3] // 2, box[2] + box[4] // 2
        page.mouse.click(x, y)
        time.sleep(0.6)
        return True
    except Exception:
        return False


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
                log(f"[camera] zoom stalled at {frac * 100:.2f}% — stopping search")
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

    if result is None:
        result = verify_view(img, self_rgb)
    log(f"[camera] FINAL: pass={result['pass']} self={result['self_frac']}% "
        f"enemies={result['enemies_visible']}")
    return result
