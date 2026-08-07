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


def leaderboard_balances(page, palette=None) -> dict:
    """OCR the leaderboard: {player_name: balance} for every row, plus the
    swatch color per row. Lets the bot know who is DRAINED (the meta kill).

    Returns (balances_by_name, swatch_by_name).
    """
    from bot.calibration import _ocr_words, swatch_from_strip
    try:
        img = grab(page)
        words = _ocr_words(img)
        if not words:
            return {}, {}
        # group into rows by y
        rows = []
        for w in words:
            t, x, y, ww, hh = w
            for r in rows:
                if abs(r["y"] - y) < 14:
                    r["words"].append(w)
                    r["y"] = min(r["y"], y)
                    break
            else:
                rows.append({"y": y, "words": [w]})
        balances, swatches = {}, {}
        for row in rows:
            ws = sorted(row["words"], key=lambda w: w[1])
            # name = leftmost word; balance = rightmost numeric word
            name = ws[0][0]
            nums = [w[0] for w in ws if w[0].isdigit()]
            bal = int(nums[-1]) if nums else None
            # swatch color: brightest saturated color left of the name
            nx = ws[0][1]
            ny = row["y"]
            nh = max((w[4] for w in ws), default=12)
            band = img[ny:ny + max(nh + 6, 12), max(0, nx - 60):max(0, nx - 2)].reshape(-1, 3).astype(int)
            sw = swatch_from_strip(band, bright_min=120)
            if bal is not None:
                balances[name] = bal
            if sw is not None:
                swatches[name] = sw
        return balances, swatches
    except Exception:
        return {}, {}


def discover_enemies(img, self_rgb, max_enemies=8):
    """Find enemy territory colors in the frame (vivid, moderate blobs, not
    terrain/UI, not our color). Needed so segment() finds attack targets."""
    from bot.calibration import saturated_colors, blob, edges_touched, dominant_colors
    H, W = img.shape[:2]
    dom = dominant_colors(img, n=6, min_frac=0.03)
    enemies = []
    for c in saturated_colors(img, max_colors=40):
        if tuple(c) == tuple(self_rgb):
            continue
        if any(np.all(np.abs(np.array(c) - np.array(d)) <= 24) for d in dom):
            continue  # terrain/UI
        b = blob(img, c, tol=20, min_area=15)
        if not b or b["area"] > 0.4 * H * W or edges_touched(b["mask"]) >= 3:
            continue
        coords = np.argwhere(b["mask"])
        if len(coords):
            if coords[:, 0].mean() > 730 or (coords[:, 0].mean() < 340 and coords[:, 1].mean() < 500):
                continue  # bottom UI / leaderboard
        enemies.append(PlayerColor(f"e{len(enemies)}", int(c[0]), int(c[1]), int(c[2])))
        if len(enemies) >= max_enemies:
            break
    return enemies


def feed_balances(page, planner, bot_name, palette=None) -> None:
    """Feed OCR'd balances into the planner: our balance (exact density) and
    enemy balances keyed by blob color (drained-target detection)."""
    balances, swatches = leaderboard_balances(page)
    if not balances:
        return
    my_bal = None
    for name, bal in balances.items():
        if name.lower() == bot_name.lower() or name.lower().startswith(bot_name.lower()[:5]):
            my_bal = bal
    if my_bal:
        planner.set_observed_balance(float(my_bal))
    # map each palette enemy (label+color) to its OCR'd balance via swatch color
    if palette and swatches and palette.enemy_colors:
        enemy_balances = {}
        for enemy in palette.enemy_colors:
            rgb = tuple(enemy.rgb)
            best_name, best_d = None, 1e9
            for name, sw in swatches.items():
                if name.lower() == bot_name.lower() or name.lower().startswith(bot_name.lower()[:5]):
                    continue
                d = (sw[0] - rgb[0]) ** 2 + (sw[1] - rgb[1]) ** 2 + (sw[2] - rgb[2]) ** 2
                if d < best_d:
                    best_d, best_name = d, name
            if best_name and best_d < 3 * 48 * 48:
                enemy_balances[enemy.name] = float(balances[best_name])
        if enemy_balances:
            planner.set_enemy_balances(enemy_balances)


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


def small_territory_colors(img, max_frac=0.03, min_area=60):
    """All saturated colors whose global blob is SMALL (a fresh territory).

    Returns [(color, blob_area)] sorted by area. Terrain/UI colors have huge
    blobs and are excluded; a just-spawned territory is small.
    """
    H, W = img.shape[:2]
    limit = max_frac * H * W
    px = img[::2, ::2].reshape(-1, 3).astype(int)
    mx = px.max(axis=1) - px.min(axis=1)
    mask = (mx > 55) & (px.max(axis=1) > 100)
    if not mask.any():
        return []
    q = (px[mask] // 16 * 16)
    out = []
    for c, _ in Counter(map(tuple, q)).most_common(40):
        area = blob_area(img, list(c), tol=20)
        if not (min_area < area < limit):
            continue
        # exclude UI zones: bottom bar (y>730) and leaderboard top-left
        m = np.all(np.abs(img.astype(int) - np.array(c)) < 20, axis=2)
        coords = np.argwhere(m)
        if len(coords):
            if coords[:, 0].mean() > 730 or (coords[:, 0].mean() < 340 and coords[:, 1].mean() < 500):
                continue
        out.append((list(c), area))
    out.sort(key=lambda x: x[1])
    return out


def detect_own_color(page, watch_s=18) -> Palette | None:
    """Click to spawn, then WATCH frames for a small territory to appear.

    Covers both cases: the click-spawn, or the game auto-placing us. Any small
    saturated blob that appears is our territory (bots' territories are bigger
    by then; terrain colors are huge). Relaxed 3% threshold — territories grow
    fast in the first seconds.
    """
    # try to spawn by double-clicking a few land spots first
    img = grab(page)
    for spot in land_spots(img, n=4):
        page.mouse.dblclick(spot[0], spot[1])
        time.sleep(0.8)
    # watch for a small territory anywhere
    deadline = time.time() + watch_s
    while time.time() < deadline:
        time.sleep(0.7)
        img = grab(page)
        cands = small_territory_colors(img)
        if cands:
            c, area = cands[0]
            log(f"SPAWN DETECTED: color={c} blob={area}")
            return Palette(self_color=PlayerColor("me", *c), enemy_colors=[],
                           tolerance=20.0, downscale=2)
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

        # 1) open the Custom Scenario editor. IMPORTANT: the main menu has a
        #    'Google Play' badge — naive text search for 'Play' would hit that.
        cs = btn("Custom Scenario")
        if cs:
            page.mouse.click(cs["x"], cs["y"])
            log(f"clicked Custom Scenario ({cs['x']},{cs['y']})")
        else:
            page.mouse.click(714, 411)
            log("clicked Custom Scenario (coords fallback)")
        time.sleep(3.5)

        # 2) reset the scenario to defaults (persistent editor state can differ
        #    between machines/sessions and shift the layout) then Play.
        reset = btn("Reset Scenario")
        if reset:
            page.mouse.click(reset["x"], reset["y"])
            log("reset scenario to defaults")
            time.sleep(3)
        else:
            log("no Reset Scenario button found")

        # verify we're really in the editor (it has a Back button); if not,
        # re-open Custom Scenario
        if not btn("Back") and not btn("Reset Scenario"):
            log("not in editor — reopening Custom Scenario")
            cs = btn("Custom Scenario")
            if cs:
                page.mouse.click(cs["x"], cs["y"])
            else:
                page.mouse.click(714, 411)
            time.sleep(3.5)

        def play_btn():
            els = page.evaluate("""() => Array.from(document.querySelectorAll('button')).map((el) => {
                const r = el.getBoundingClientRect();
                return {text: (el.innerText || '').trim(), x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2)};
            })""")
            # STRICT: strip emojis then exact-match 'Play'. '⚔️ Multiplayer'
            # would otherwise match a loose 'starts with ⚔️' check.
            for e in els:
                t = e["text"].replace("⚔️", "").replace("🗡️", "").strip()
                if t == "Play":
                    return e
            return None

        play = play_btn()
        if not play:
            page.keyboard.press("Escape")
            time.sleep(1.5)
            play = play_btn()
        if play:
            page.mouse.click(play["x"], play["y"])
            log(f"clicked Play ({play['x']},{play['y']})")
        else:
            log("NO Play button — aborting")
            browser.close()
            sys.exit(1)
        time.sleep(4)

        # 3) detect own color via the spawn spot (autonomous, no OCR)
        palette = detect_own_color(page)
        if palette is None:
            log("no spawn detected after retries — saved frame")
            snapshot(page, "calib_fail")
            browser.close()
            sys.exit(1)
        # discover enemy colors so segment() finds attack targets
        img = grab(page)
        enemy_colors = discover_enemies(img, tuple(palette.self_color.rgb))
        log(f"discovered {len(enemy_colors)} enemy colors: "
            f"{[tuple(e.rgb) for e in enemy_colors]}")
        palette = Palette(self_color=palette.self_color, enemy_colors=enemy_colors,
                          tolerance=24.0, downscale=2)

        # 4) play with the trained combat brain
        planner = make_planner()
        controls = MouseControls(page)
        report = {"areas": [], "eliminations": []}
        action_counts: dict[str, int] = {}
        start = time.time()
        last_shot = time.time()
        last_ocr = time.time()
        last_enemy_colors: set = set()   # track COLORS (labels get reassigned)
        seen_counts: dict = {}           # stability: colors seen steadily
        miss_counts: dict = {}           # colors missing steadily -> eliminated
        eliminated = 0
        log(f"BOT PLAYING for {PLAY_MINUTES} min...")
        while time.time() - start < PLAY_MINUTES * 60:
            try:
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
                    else:
                        log(">>> WE WERE ELIMINATED")
                        break
                except Exception:
                    pass
                # feed live balances (exact density + drained-enemy targeting)
                # and rediscover enemy colors (they appear/grow over time)
                if time.time() - last_ocr > 6:
                    try:
                        feed_balances(page, planner, BOT_NAME, palette)
                    except Exception as e:
                        log(f"  balance OCR err: {e}")
                    try:
                        img2 = grab(page)
                        en = discover_enemies(img2, tuple(palette.self_color.rgb))
                        now_colors = {tuple(e.rgb) for e in en}
                        # stability-based elimination: a color seen steadily
                        # (>=2 checks) then missing steadily (>=2 checks) is a
                        # real elimination — flicker/relabel is filtered out
                        for c in now_colors:
                            seen_counts[c] = seen_counts.get(c, 0) + 1
                            miss_counts.pop(c, None)
                        for c in list(seen_counts):
                            if c not in now_colors:
                                miss_counts[c] = miss_counts.get(c, 0) + 1
                                if miss_counts[c] >= 2 and seen_counts[c] >= 2:
                                    eliminated += 1
                                    log(f">>> ENEMY ELIMINATED {list(c)} (total {eliminated})")
                                    report["eliminations"].append({"t": round(time.time() - start, 1),
                                                                   "color": [int(v) for v in c]})
                                    del seen_counts[c]
                                    del miss_counts[c]
                            else:
                                miss_counts[c] = 0
                        last_enemy_colors = now_colors
                        if en:
                            palette = Palette(self_color=palette.self_color, enemy_colors=en,
                                              tolerance=24.0, downscale=2)
                    except Exception as e:
                        log(f"  rediscover err: {e}")
                    last_ocr = time.time()
                if time.time() - last_shot > 5:
                    snapshot(page)
                    last_shot = time.time()
                # incremental report so a crash never loses the data
                report["max_area"] = max([a["area"] for a in report["areas"]], default=0)
                report["actions"] = dict(sorted(action_counts.items(), key=lambda kv: -kv[1]))
                with open(os.path.join(OUT, "battle_report.json"), "w") as f:
                    json.dump(report, f, indent=2)
            except Exception as e:
                log(f"loop error (continuing): {e}")
                time.sleep(1)

        report["max_area"] = max([a["area"] for a in report["areas"]], default=0)
        report["actions"] = dict(sorted(action_counts.items(), key=lambda kv: -kv[1]))
        # last-survivor check: no enemies detected in the final frame
        try:
            final_img = grab(page)
            enemies_left = discover_enemies(final_img, tuple(palette.self_color.rgb))
            report["last_survivor"] = len(enemies_left) == 0
        except Exception:
            report["last_survivor"] = False
        log("===== BATTLE REPORT =====")
        log(json.dumps(report, indent=2))
        with open(os.path.join(OUT, "battle_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        browser.close()


if __name__ == "__main__":
    main()
