#!/usr/bin/env python3
"""Territorial.io bot — FULLY AUTONOMOUS, no manual steps.

The bot:
  1. opens territorial.io in headless Chromium
  2. (best-effort) picks a vivid color in the editor so it's easy to track
  3. joins Custom Scenario -> Play
  4. double-clicks land spots and DIFF-DETECTS its own spawn color
     (a new small colored blob appears where it clicks — causal, no OCR)
  5. plays with the TRAINED brain (evolved weights: 8/8 wins vs 3 bots
     in the offline simulator) + MouseControls
  6. writes battle_report.json + frames + bot.log

Run:  python run_bot.py   (config via env vars, see CONFIG below)
"""
import sys, os, io, time, json
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- CONFIG ---
GAME_URL = "https://territorial.io/"
BOT_NAME = os.environ.get("BOT_NAME", "AureliaBot")
PLAY_MINUTES = float(os.environ.get("PLAY_MINUTES", "4"))
DECISION_HZ = float(os.environ.get("DECISION_HZ", "2.5"))  # click decisions per second
MAX_FRAMES_KEEP = 60
_MANUAL = os.environ.get("MANUAL_COLOR", "").strip()
MANUAL_COLOR = json.loads(_MANUAL) if _MANUAL else None  # optional override
SELF_RGB = [255, 60, 60]  # preferred vivid color (best-effort editor set)
# ---------------------------------------------------------------------------

OUT = os.environ.get("KAGGLE_WORKING", "/kaggle/working") if os.path.isdir("/kaggle/working") else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bot_output")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "bot.log")

from playwright.sync_api import sync_playwright

from bot.config import Palette, PlayerColor, LoopConfig
from bot.planner import ClickPlanner, ClickPlannerConfig
from bot.economy import TroopTracker
from bot.controls import MouseControls
from bot.click_loop import ClickLoop

FLAGS = [
    "--no-sandbox", "--disable-dev-shm-usage", "--use-gl=swiftshader", "--disable-gpu",
    "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
    "--disable-background-timer-throttling", "--disable-background-video-track-optimizations",
    "--disable-features=CalculateNativeWinOcclusion", "--enable-unsafe-swiftshader",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


shot_counter = 0


def snapshot(page, name="frame") -> str:
    global shot_counter
    shot_counter += 1
    path = os.path.join(OUT, f"{name}_{shot_counter:04d}.png")
    page.screenshot(path=path)
    import glob
    for old in sorted(glob.glob(os.path.join(OUT, "frame_*.png")))[:-MAX_FRAMES_KEEP]:
        os.remove(old)
    return path


def grab(page) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))


def land_spots(img, n=10) -> list[tuple[int, int]]:
    """Candidate land pixels (not ocean, not leaderboard/bottom UI)."""
    px = img.astype(int)
    r, g, b = px[:, :, 0], px[:, :, 1], px[:, :, 2]
    is_land = ~((b > r + 20) & (b > 80)) & (px.max(axis=2) >= 60)
    # avoid the leaderboard (top-left) and bottom UI strip
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    ok = is_land & (yy > 320) & (xx > 500) & (yy < 740)
    ys, xs = np.where(ok)
    if not len(ys):
        ys, xs = np.where(is_land)
    if not len(ys):
        return [(640, 400)] * n
    rng = np.random.default_rng(3)
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]


def spot_color(img, spot) -> list[int]:
    """Mean color of a small patch at the spot."""
    x, y = spot
    h, w = img.shape[:2]
    patch = img[max(0, y - 3):min(h, y + 4), max(0, x - 3):min(w, x + 4)].reshape(-1, 3).astype(int)
    return patch.mean(axis=0).astype(int).tolist()


def blob_area(img, color, tol=24) -> int:
    m = np.all(np.abs(img.astype(int) - np.array(color)) < tol, axis=2)
    return int(m.sum())


def detect_own_color(page, attempts=10) -> Palette | None:
    """Double-click land spots; the spawn makes a SMALL colored blob appear at
    the click point. Sample the click-point color at 0.4s — if its global blob
    is small (<2% frame), that's OUR territory color (causal, no OCR)."""
    img = grab(page)
    spots = land_spots(img)
    for spot in spots:
        page.mouse.dblclick(spot[0], spot[1])
        log(f"double-clicked ({spot[0]},{spot[1]})")
        time.sleep(0.4)
        img = grab(page)
        c = spot_color(img, spot)
        b = blob_area(img, c, tol=24)
        log(f"  spot color={c} blob={b}")
        if 8 < b < 0.02 * 1280 * 800:
            log(f"SPAWN DETECTED at {spot}: color={c} blob={b}")
            return Palette(self_color=PlayerColor("me", *c), enemy_colors=[],
                           tolerance=24.0, downscale=2)
        # clear any selection state, try the next spot
        page.keyboard.press("Escape")
        time.sleep(0.4)
    return None


def set_own_color(page) -> bool:
    """Best-effort: pick SELF_RGB in the editor (vivid = easy to track)."""
    try:
        page.mouse.click(714, 411)   # Custom Scenario
        time.sleep(3.5)
        page.mouse.click(1115, 242)  # Colors Settings
        time.sleep(3)
        page.mouse.click(50, 204)    # Customized
        time.sleep(3)
        for y, val in [(187, SELF_RGB[0]), (211, SELF_RGB[1]), (235, SELF_RGB[2])]:
            page.mouse.click(351, y)
            time.sleep(0.6)
            page.keyboard.press("Control+A")
            page.keyboard.type(str(val))
            time.sleep(0.4)
            page.keyboard.press("Enter")
            time.sleep(1.0)
        page.mouse.click(640, 759)   # Back to editor
        time.sleep(2)
        log(f"set own color to {SELF_RGB}")
        return True
    except Exception as e:
        log(f"set_own_color failed: {e}")
        return False


def load_weights() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best_weights.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def make_planner() -> ClickPlanner:
    cfg = ClickPlannerConfig()
    for k, v in load_weights().items():
        if hasattr(cfg, k):
            setattr(cfg, k, float(v))
    return ClickPlanner(cfg, TroopTracker(balance=512.0, land=12))


def main() -> None:
    with open(LOG, "w") as f:
        f.write("bot session\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=FLAGS)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        page.goto(GAME_URL, timeout=60000, wait_until="domcontentloaded")
        time.sleep(7)
        page.bring_to_front()
        try:
            page.fill('input[placeholder*="Kingdom"], input', BOT_NAME)
        except Exception:
            pass
        time.sleep(1)

        def btn(text):
            els = page.evaluate("""() => Array.from(document.querySelectorAll('button')).map((el) => {
                const r = el.getBoundingClientRect();
                return {text: (el.innerText || '').trim(), x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2)};
            })""")
            return next((e for e in els if text in e["text"]), None)

        # 1) best-effort vivid color (does NOT block if it fails)
        set_own_color(page)

        # 2) Play (may need Escape if a panel is still open)
        play = btn("Play")
        if not play:
            page.keyboard.press("Escape")
            time.sleep(1.5)
            play = btn("Play")
        if play:
            page.mouse.click(play["x"], play["y"])
            log(f"clicked Play ({play['x']},{play['y']})")
        else:
            log("NO Play button — aborting")
            browser.close()
            sys.exit(1)
        time.sleep(6)

        # 3) detect own color via spawn diff (autonomous, no OCR)
        palette = detect_own_color(page)
        if palette is None:
            log("no spawn detected after retries — saved frame")
            snapshot(page, "calib_fail")
            browser.close()
            sys.exit(1)

        # 4) play with the trained brain
        planner = make_planner()
        controls = MouseControls(page)
        report = {"areas": []}
        action_counts: dict[str, int] = {}
        start = time.time()
        last_shot = time.time()
        log(f"BOT PLAYING for {PLAY_MINUTES} min...")
        while time.time() - start < PLAY_MINUTES * 60:
            loop = ClickLoop(capture=lambda: grab(page), palette=palette, brain=planner,
                             controls=controls, loop_cfg=LoopConfig(hz=DECISION_HZ),
                             decision_interval_s=1.0 / DECISION_HZ)
            stats = loop.run(duration_s=8, max_ticks=int(DECISION_HZ * 8))
            for k, v in stats.snapshot()["actions"].items():
                action_counts[k] = action_counts.get(k, 0) + v
            img = grab(page)
            try:
                from bot.vision import segment
                st = segment(img, palette)
                if st.self_blob:
                    report["areas"].append({"t": round(time.time() - start, 1),
                                            "area": st.self_blob.area})
            except Exception:
                pass
            if time.time() - last_shot > 5:
                snapshot(page)
                last_shot = time.time()

        report["max_area"] = max([a["area"] for a in report["areas"]], default=0)
        report["actions"] = dict(sorted(action_counts.items(), key=lambda kv: -kv[1]))
        log("===== BATTLE REPORT =====")
        log(json.dumps(report, indent=2))
        with open(os.path.join(OUT, "battle_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        browser.close()


if __name__ == "__main__":
    main()
