#!/usr/bin/env python3
"""Territorial.io bot — FULLY AUTONOMOUS.

The bot:
  1. opens territorial.io in headless Chromium
  2. chooses its OWN vivid color in the Custom Scenario editor (no OCR needed)
  3. loads TRAINED weights (weights/best_weights.json — evolved in the offline
     simulator to maximize win rate vs bots)
  4. joins Custom Scenario -> Play, double-clicks a start position
  5. plays with ClickPlanner + MouseControls (double-click to expand/attack)
  6. writes battle_report.json + frames + bot.log

Run:  python run_bot.py   (config via env vars, see CONFIG below)
"""
import sys, os, io, time, json

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

# The bot's own vivid color, chosen in the editor (vivid red — never on maps).
SELF_RGB = [255, 60, 60]
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
from bot.calibration import calibrate_from_leaderboard, center_lock_calibrate

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


def land_spot(img) -> tuple[int, int]:
    """A clickable land pixel (not ocean) for the start-position double-click."""
    px = img.astype(int)
    r, g, b = px[:, :, 0], px[:, :, 1], px[:, :, 2]
    is_land = ~((b > r + 20) & (b > 80)) & (px.max(axis=2) >= 60)
    ys, xs = np.where(is_land)
    return (int(xs[len(ys) // 2]), int(ys[len(ys) // 2])) if len(ys) else (640, 400)


# ==== 1. SELF-COLOR: pick our own vivid color in the editor ==================
# Editor layout (verified live): Custom Scenario -> Colors Settings (1115,242)
# -> Customized (50,204) -> RGB fields at (351,187)/(351,211)/(351,235).
def set_own_color(page) -> bool:
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
        log(f"set own color to {SELF_RGB}")
        return True
    except Exception as e:
        log(f"set_own_color failed: {e}")
        return False


def calibrate(page, know_color=None) -> Palette | None:
    """We KNOW our color (self-chosen); this is just a sanity fallback chain."""
    if know_color is not None:
        return Palette(self_color=PlayerColor("me", *know_color), enemy_colors=[],
                       tolerance=24.0, downscale=2)
    if MANUAL_COLOR is not None:
        return Palette(self_color=PlayerColor("me", *MANUAL_COLOR), enemy_colors=[],
                       tolerance=24.0, downscale=2)
    log("self-color not set — falling back to leaderboard OCR")
    for attempt in range(4):
        img = grab(page)
        pal, reason = calibrate_from_leaderboard(img, BOT_NAME)
        if pal:
            log(f"CALIBRATED via leaderboard ({reason})")
            return pal
        log(f"leaderboard calib attempt {attempt}: {reason}")
        pal2 = center_lock_calibrate(page, lambda: grab(page), timeout_s=15)
        if pal2:
            log("CALIBRATED via center-lock")
            return pal2
    return None


# ==== 2. TRAINED WEIGHTS: evolve ClickPlannerConfig in the offline sim =======
def load_weights() -> dict:
    """Load trained weights (JSON) if present; {} otherwise."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best_weights.json")
    try:
        with open(path) as f:
            w = json.load(f)
        log(f"loaded trained weights: {w}")
        return w
    except Exception:
        log("no weights file — using defaults")
        return {}


def make_planner() -> ClickPlanner:
    cfg = ClickPlannerConfig()
    weights = load_weights()
    for k, v in weights.items():
        if hasattr(cfg, k):
            setattr(cfg, k, float(v))
    return ClickPlanner(cfg, TroopTracker(balance=512.0, land=12))


# ==== 3. MAIN =================================================================
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

        # 1) choose our own color (also lands us in the editor)
        ok_color = set_own_color(page)
        if not ok_color:
            # try again via DOM buttons
            cs = btn("Custom Scenario")
            if cs:
                page.mouse.click(cs["x"], cs["y"])
                time.sleep(3)
        time.sleep(1)

        # 2) Play
        play = btn("Play")
        if play:
            page.mouse.click(play["x"], play["y"])
            log(f"clicked Play ({play['x']},{play['y']})")
        else:
            log("NO Play button — aborting")
            browser.close()
            sys.exit(1)
        time.sleep(6)

        # 3) choose start position (double-click land)
        img = grab(page)
        spot = land_spot(img)
        page.mouse.dblclick(spot[0], spot[1])
        log(f"double-clicked start position at {spot}")
        time.sleep(2)

        # 4) palette: we KNOW our color
        palette = calibrate(page, know_color=SELF_RGB if ok_color else None)
        if palette is None:
            log("calibration failed — saved frame")
            snapshot(page, "calib_fail")
            browser.close()
            sys.exit(1)

        # 5) play with trained weights
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
