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

## Run on Kaggle

Upload `kaggle_runner.ipynb` (in the workspace) or use this:

```python
!git clone https://github.com/amerameryou1-blip/bot.git /kaggle/working/bot
!pip install -q playwright pillow numpy pytesseract
!apt-get install -y -qq tesseract-ocr
!playwright install chromium
!playwright install-deps chromium
!python /kaggle/working/bot/run_bot.py
```

Results: `/kaggle/working/battle_report.json` + `frame_*.png` + `bot.log`.

## Config

Edit the top of `run_bot.py`:

| var | meaning |
| --- | --- |
| `BOT_NAME` | your in-game name (keep distinctive for calibration) |
| `PLAY_MINUTES` | match length (matches run 3–8 min, no auto-restart) |
| `DECISION_HZ` | click decisions per second |
| `MANUAL_COLOR` | your RGB if you set a color in the editor (skips OCR) |

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
