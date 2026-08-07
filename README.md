# 🤖 Territorial.io Bot

A lightweight AI bot for territorial.io **Custom Scenario** (single-player vs bots).
The game works on **double-click**: double-click land to expand, set the attack-%
slider (W/S/D/A) and double-click an enemy border to attack.

- **Vision** — color segmentation → territory map + expand/attack click targets
- **Brain** — `ClickPlanner`: economy-aware (interest 0.56s, income 5.6s, soft/hard
  density limits, 2:1 defender advantage), expands at low %, exploits exhausted neighbors
- **Calibration** — automatic per-match color from the leaderboard swatch (OCR);
  colors change every match, so nothing is hardcoded
- **Offline** — 42 unit tests + a click simulator (`scripts/click_tournament.py`)

## Run on Kaggle — ONE CELL

```python
import os
os.environ['BOT_NAME'] = 'AureliaBot'      # your in-game name
os.environ['PLAY_MINUTES'] = '4'          # match length (3-8 is normal)
# os.environ['MANUAL_COLOR'] = '[255, 60, 60]'   # <-- RELIABLE: set this!

!git clone -q https://github.com/amerameryou1-blip/bot.git /kaggle/working/bot
!pip install -q playwright pillow numpy pytesseract
!apt-get install -y -qq tesseract-ocr > /dev/null 2>&1 || true
!playwright install chromium
!playwright install-deps chromium
!python /kaggle/working/bot/run_bot.py
```

Results: `/kaggle/working/battle_report.json` + `frame_*.png` + `bot.log`.

## ⭐ MANUAL_COLOR — the reliable way (read this)

The game assigns you a **random color each match**, and it can pick a color that
**matches the map terrain** — which breaks pixel-vision calibration (verified
live: the bot got dark/terrain-matching colors in 3 of 3 test matches). The
100% reliable fix:

1. In the **Custom Scenario editor**, pick a **vivid color** for yourself
   (Settings → Colors → a bright red/orange/blue/purple — something not on the map).
2. Set `MANUAL_COLOR = "[r, g, b]"` (or the `os.environ` line above) to that color.

Then calibration is instant, exact, and immune to OCR/terrain issues.

## Config

Edit the top of `run_bot.py` (or env vars):

| var | meaning |
| --- | --- |
| `BOT_NAME` | your in-game name |
| `PLAY_MINUTES` | match length (matches run 3–8 min, no auto-restart) |
| `DECISION_HZ` | click decisions per second |
| `MANUAL_COLOR` | ⭐ your RGB — set this for reliable play |

## Live-test status (via Kaggle API — 5 automated cycles)

The push→auto-run→pull loop works end-to-end (Kaggle API). Five live matches
were run and analyzed. The honest finding:

- The bot reliably: opens the game, joins Custom Scenario, double-clicks a
  start position, plays, and reports.
- **The blocker: the game assigns the player a random color each match, and in
  our test matches it kept assigning colors that match the map terrain or the
  leaderboard UI** (dark greens, teal UI, yellow-green terrain). Pixel-vision
  cannot separate "my 12-pixel territory" from identical-colored terrain/UI.
- The calibration now: (1) samples the color at the double-click spawn spot
  (causal — rejects blobs >2% of frame), (2) falls back to the leaderboard
  swatch (brightness-filtered, whole-row scan), (3) **fails fast with a clear
  message** instead of playing blind.

**Why MANUAL_COLOR is the answer:** if you pick a vivid color in the editor
(e.g. pure red) the map never contains it, so calibration is instant and
exact. One 30-second manual step makes the bot reliable.

## Local dev

```bash
PYTHONPATH=src pytest tests -q                      # 42 tests
PYTHONPATH=src python3 scripts/click_tournament.py  # offline win-rate
python3 run_bot.py                                  # play a match locally
```

## Notes

- Multiplayer lobby is often unreachable from datacenter IPs; Custom Scenario
  runs the match in-browser and avoids it.
- Headless Chromium only repaints the canvas after input events → the bot
  screenshots right after its own clicks.
- Botting may violate territorial.io's ToS — use in bot rooms / casually.
