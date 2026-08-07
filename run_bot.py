#!/usr/bin/env python3
"""Territorial.io bot — run this on Kaggle (or anywhere) to play a match.

The game works on DOUBLE-CLICK (confirmed by the player): double-click land to
expand, set the attack-% slider (W/S/D/A) and double-click an enemy border to
attack. This script:
  1. opens territorial.io in headless Chromium
  2. Custom Scenario -> Play
  3. double-clicks a start position
  4. calibrates YOUR color (leaderboard swatch OCR, or MANUAL_COLOR)
  5. plays with ClickPlanner + MouseControls for PLAY_MINUTES
  6. writes battle_report.json + frames

Run:  python run_bot.py
"""
import sys, os, io, time, json

# allow `python run_bot.py` from anywhere in the repo
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
MANUAL_COLOR = json.loads(_MANUAL) if _MANUAL else None  # e.g. "[242, 216, 63]"

# output dir: Kaggle working dir if present, else ./bot_output
OUT = os.environ.get("KAGGLE_WORKING", "/kaggle/working") if os.path.isdir("/kaggle/working") else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bot_output")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "bot.log")
# ---------------------------------------------------------------------------

from playwright.sync_api import sync_playwright

from bot.config import Palette, PlayerColor, LoopConfig
from bot.planner import ClickPlanner, ClickPlannerConfig
from bot.economy import TroopTracker
from bot.controls import MouseControls
from bot.click_loop import ClickLoop
from bot.calibration import calibrate_from_leaderboard, center_lock_calibrate, _ocr_words, find_name_box

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


def ocr_balance(page) -> int | None:
    """Best-effort: my balance from the leaderboard row (OCR)."""
    try:
        img = grab(page)
        words = _ocr_words(img)
        box, _ = find_name_box(words, BOT_NAME)
        if box is None:
            return None
        _, nx, ny, nw, nh = box
        row = [w for w in words if abs(w[2] - ny) < 12 and w[1] > nx + nw]
        nums = [int(w[0]) for w in row if w[0].isdigit()]
        return nums[0] if nums else None
    except Exception:
        return None


def calibrate(page) -> Palette | None:
    if MANUAL_COLOR is not None:
        log(f"using MANUAL_COLOR {MANUAL_COLOR}")
        return Palette(self_color=PlayerColor("me", *MANUAL_COLOR), enemy_colors=[],
                       tolerance=48.0, downscale=2)
    log("auto-calibrating from leaderboard...")
    log("  TIP: for reliable play, pick a VIVID color (red/orange/blue/purple) in the")
    log("       Custom Scenario editor, then set MANUAL_COLOR = [r, g, b] in run_bot.py")
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

        def btn(text):
            els = page.evaluate("""() => Array.from(document.querySelectorAll('button')).map((el) => {
                const r = el.getBoundingClientRect();
                return {text: (el.innerText || '').trim(), x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2)};
            })""")
            return next((e for e in els if text in e["text"]), None)

        cs = btn("Custom Scenario")
        if cs:
            page.mouse.click(cs["x"], cs["y"])
            log("clicked Custom Scenario")
        time.sleep(4)
        play = btn("Play")
        if play:
            page.mouse.click(play["x"], play["y"])
            log(f"clicked Play ({play['x']},{play['y']})")
        else:
            log("NO Play button — aborting")
            browser.close()
            sys.exit(1)
        time.sleep(6)

        # ---- start position: double-click land, retry a few spots ----
        palette = None
        for spot_attempt in range(6):
            img = grab(page)
            spot = land_spot(img)
            # vary the spot a bit each attempt in case the first was water/blocked
            if spot_attempt > 0:
                h, w = img.shape[:2]
                spot = (min(w - 10, spot[0] + 60 * (spot_attempt % 3) - 60),
                        min(h - 10, spot[1] + 40 * ((spot_attempt // 3) % 2) - 20))
            page.mouse.dblclick(spot[0], spot[1])
            log(f"double-clicked start position at {spot}")
            time.sleep(5)
            palette = calibrate(page)
            if palette:
                break
            snapshot(page, "calib_fail")  # keep a frame so the user can see
        if palette is None:
            log("calibration failed after retries — frames saved as calib_fail_*")
            browser.close()
            sys.exit(1)

        # ---- play ----
        planner = ClickPlanner(ClickPlannerConfig(), TroopTracker(balance=512.0, land=12))
        controls = MouseControls(page)
        report = {"areas": []}
        action_counts: dict[str, int] = {}
        start = time.time()
        last_shot = time.time()
        last_ocr = time.time()
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
            if time.time() - last_ocr > 6:
                bal = ocr_balance(page)
                if bal:
                    planner.set_observed_balance(bal)
                    log(f"  OCR balance={bal}")
                last_ocr = time.time()
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
