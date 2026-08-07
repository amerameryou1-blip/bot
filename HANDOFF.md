# 🔁 HANDOFF — Next Agent (you have VISION — use it)

> Written 2026-08-07 by the previous agent, for the agent taking over this project.
> Goal: get the territorial.io bot to **win as LAST SURVIVOR** (not biggest area) in the **real game**, trained on **real data**.
> This doc tells you exactly what's built, what's broken, what the user wants, and what to do next.

---

## 0. READ THIS FIRST — the current blocker (why you're here)

**The bot plays the real game but the camera is zoomed in on its own spawn.**
It cannot see the map, cannot see enemies, so it never attacks → it records useless
data and never wins as last survivor. **This is THE problem to solve.**
The user (a real person who plays territorial.io) said:

> "when you choose spawn spot it zooms in to your location... you need to zoom out
> or zoom a reasonable amount so the bot can see / record other bots playing"

**You have vision — LOOK at the recorded frames to confirm:**
`amer224/territorial-bot-data/recordings/20260807-173018-492c50/frames/` (123 frames).
They are ~90% dark fog-of-war with one small lit patch around our spawn — that's the
zoomed-in camera. The bot's own `detect_own_color()` saw our blob shrink from 0.3% →
0.01% of screen as it (tried to) grow — no, the CAMERA stayed zoomed on a tiny area.

### What was tried (and failed) for zoom, before you re-investigate
- `page.mouse.wheel(0, -800)` / `+800` — no effect (game ignores synthetic wheel deltas).
- Keyboard `+`/`-`/`=`/`_`/`0` — no effect.
- Clicking the on-screen zoom buttons — not found (they're canvas-rendered).
- The game JS (inline script, ~657KB) has real zoom internals:
  - `du.zoom(a3, aD, aE)` — core zoom fn, `hs*=ft` multiplier.
  - `"Zoom In"` / `"Zoom Out"` are **reassignable key actions** in the settings list
    (the game lets users bind keys, like "Hide UI" etc.).
  - There's a leaderboard name-click that **centers the camera on you** (the user told us:
    clicking your name in the leaderboard recenters) — but that re-centers, not zoom-out.

### The user's suggested trick (you have vision → you can verify)
> "you can fetch your name from the leaderboard, your name is always visible; when
> clicked it takes you back to your location; make sure to zoom out so the bot can see"

So the plan the user expects: **OCR/click your name in the leaderboard → camera centers
on you → then zoom out** so the bot sees a reasonable area (ideally most of the map or
at least a large region with enemies).

### Ideas you should try FIRST (vision makes this way easier)
1. **Look at a live frame with your vision.** Start the bot, screenshot, and *see* the
   screen. That alone will reveal the zoom UI (buttons in a corner?), the camera
   behavior, and the fog-of-war pattern.
2. **Find the real zoom input.** Inspect the game's inline JS for how wheel/keyboard
   zoom is dispatched (search `"Zoom In"`, `"Zoom Out"`, `wheel`, `du.zoom`). The keys
   might be **not bound by default** → try dispatching real trusted events, or find the
   default keybinding map in the JS (search `aAG(15,"zoom"...` — that's an icon, but
   the key binding table is nearby).
3. **Zoom buttons**: the game draws zoom +/- buttons on canvas (the JS references
   `"Hide Zoom Buttons"` in settings → there ARE zoom buttons by default). Find their
   screen coordinates by color/location (bottom-right corner is typical) and click them.
4. **Leaderboard name click**: implement `calibrate_from_leaderboard` + find your row,
   click the name → camera centers → then zoom out via (1-3).
5. **The nuclear option**: find the internal zoom variable in JS and set it directly via
   the page console (e.g. patch the game's zoom state), or dispatch a real `wheel` event
   with `{bubbles:true, deltaY:-1000}` on the canvas element itself.

**Success criterion:** a recorded frame where our territory is visible AND a meaningful
portion of the map (enemies visible), e.g. our blob ≥ 1–5% of screen and ≥ 2 enemy
colors present outside the leaderboard.

---

## 1. The user's goal (verbatim intent, synthesized)

- A bot that plays **territorial.io** for real (in a browser) and **wins as last survivor**.
- It must **see the screen (CNN)**, know **which pixels are "me"**, know **water vs land**,
  and **attack properly (the meta)** — expand to free land, then attack weak enemies.
- Training on **real data first, highest quality over quantity**: real screenshots, real
  recordings, real clicks, multiple enemies, different colors, real maps.
- The user is moving me (previous agent, no vision) → **you (with vision) to make it
  much easier**. Use vision aggressively: look at frames, screenshots, the game UI.
- The user has ~**36 GPU-hours/week on Kaggle** (5 concurrent CPU sessions), models go to
  **Hugging Face**.
- User wants **loud failures** — never a silent "COMPLETE" that didn't learn.
- User will **rotate all credentials after the project** — treat them as temporary.

---

## 2. 🔑 CREDENTIALS (user-provided; they will rotate them — do not hardcode into public files)

| Service | Username | Token | Where used |
|---|---|---|---|
| GitHub | `amerameryou1-blip` | `<GITHUB_TOKEN>` | pushing `bot` repo (public) |
| Kaggle | `amerameryou` | file `~/.kaggle/access_token` = `<KAGGLE_TOKEN>` | Kaggle API / kernels |
| Hugging Face | `amer224` | `<HF_TOKEN>` | HF datasets + model repos |

Git identity for this sandbox (resets each session):
```bash
git config user.email "amerameryou1-blip@users.noreply.github.com"
git config user.name "amerameryou1-blip"
```
Push command (uses token inline):
```bash
git push https://<GITHUB_TOKEN>@github.com/amerameryou1-blip/bot.git main
```

---

## 3. Where everything lives

### GitHub (public, user reads README as status dashboard)
- Repo: `amerameryou1-blip/bot` — the main code repo (mirrors `/home/user/bot-repo`).
- **README.md is the user's dashboard** — keep it updated with run status + session table.

### Hugging Face (all private)
- `amer224/territorial-bot-data` (dataset, private):
  - `screenshots/<map>/` — 26 real start-phase screenshots, 23 maps, organized,
    with `index.json` (map name, real dimension, OCR'd land/water/mountain %).
  - `shard_*.npz` — sim data (~9,443 samples across island/mountains/desert/swamp/lakes).
  - `recordings/20260807-173018-492c50/` — the ONE real match recording so far
    (123 frames + clicks + meta). **This is the zoomed-in junk** — useful only for
    proving the problem; new recordings should replace it.
- `amer224/territorial-bot-nn` (model repo, private, mostly empty) — where trained
  weights go via `scripts/export_hf.py`.

### Local sandbox workspace (`/home/user`)
- `bot-repo/` — the code (synced to GitHub).
- `kaggle-push/` — Kaggle notebooks: `kaggle_train_nn.ipynb` (main GPU trainer),
  `worker_{island,mountains,desert,swamp}.ipynb` (CPU data collectors, done),
  `kernel-metadata.json`.
- `realdata/shots/` — the 26 screenshots (local copy, 13MB — same as HF).
- `datacheck/`, `uploads/` (user's custom-scenario setup screenshot), `territorial-bot/` (OLD
  deprecated project — has `scripts/tune.py`, `tournament.py` etc., mostly superseded).

---

## 4. The codebase (bot-repo)

### Live bot — `run_bot.py` (plays the real game via Playwright headless Chromium)
- Opens territorial.io, clicks **Custom Scenario → Reset Scenario → Play** (strict
  "Play" matcher that strips emoji to avoid matching "Best 1v1 Play-er"/"Google Play").
- `detect_own_color()`: double-clicks land spots and **diff-detects** the new small
  saturated blob that appears where we spawn (causal, no OCR) — works.
- `discover_enemies()`: finds vivid saturated color blobs outside UI zones.
- `feed_balances()`: OCRs leaderboard balances to know enemy troop strength.
- Plays via `ClickLoop` (capture→segment→planner.decide→mouse) at `DECISION_HZ` (2.5/s).
- **`--record --games N --upload`** → records every frame + click to
  `recordings/<session>/` and uploads to HF. (This is how we get real data.)
- **KNOWN BUG**: camera zooms into spawn (see §0). Also the bot "expanded but never
  attacked" in its one real win (rank #1 by area, not last survivor).

### Sim — `src/sim/game6.py` (primary, v8) and `game5.py` (older)
- game6: **real maps** from `weights/maps/*.npz` (extracted from the 26 screenshots),
  8–15 enemies, mixed-skill lobbies, vivid per-player colors, kill tracking, water &
  mountain impassable. Win = last survivor. Same interface as game5.
- game5: procedural maps (lakes/island/mountains/desert/swamp), 3-4 bots.

### Real maps — `scripts/rebuild_maps.py`
- Extracts coastline/water/mountain masks from the 26 start screenshots, validates
  land/water/mountain % against OCR'd in-game stats (±5%). **7 maps PASS**: island,
  desert, pond, island_kingdom, middle_east, white_arena, black_arena. 19 FAIL loudly
  (ambiguous gray land vs UI). Preview PNGs + ascii maps in `weights/maps/`.

### NN — `src/nn/model.py` (TerritoryNet, 85,533 params, CPU-fast)
- 64×64 RGB in → 4 conv blocks → (64,16,16) features → heads:
  segmentation (5 classes: water/neutral/me/enemy/ui), localization, click-map (16×16),
  kind (expand/attack/bank), attack %, value (RL).
- `src/nn/data.py`, `src/nn/bot_brain.py`.

### Trainer — `scripts/train_nn.py`
- Stages: `collect` (teacher sim data), `vision` (seg+localize), `clone` (click head),
  `real` (fine-tune on real frames + clicks — **added recently, needs real data**),
  `ppo` (rewards: growth/2000 + kill×2 + win +5 − idle penalty; curriculum escalates
  **skill AND enemy count** 2→…→10), `eval`.
- P100 CUDA fix: auto-installs torch 2.4.1+cu121; `_safe_device()` actually tests ops;
  FORCE_CPU=1 fallback.
- **v7 lesson**: survival-only reward + medium-only curriculum → alive 0.00 for 100
  rounds (learned nothing). v8's win/kill rewards + easy-start + enemy-count curriculum
  is the fix — PPO smoke test passed locally (alive 1.00, curriculum fired).

### Auto-labeler — `scripts/label_real.py`
- Turns recordings + screenshots into 5-class per-pixel labels + click targets.
  Verified locally on fake sessions. Needs real recordings to be useful.

### Others
- `src/bot/` — planner (meta teacher), economy (TroopTracker, interest/income ticks),
  vision (segment/find targets), controls, click_loop, calibration (OCR leaderboard,
  swatch reading — **`calibrate_from_leaderboard()` exists = the name-click trick's
  foundation**), config, state, loop, strategy, **recorder.py**.
- `weights/best_weights.json` — evolved heuristic weights (8/8 vs old sim bots).
- `weights/nn/model.pt` — last trained model (v7 result: survives easy, never wins).
- `tests/` — 47 tests pass (`PYTHONPATH=src python3 -m pytest tests -q`).
- `QUALITY_PLAN.md` — the quality-first roadmap (data pyramid, ship gates).
- `MECHANICS.md`, `ATTACK_META.md`, `QUESTIONS.md` — game knowledge.

---

## 5. Game knowledge (from the user + live testing)

- Controls: **double-click claims land** (canvas-only, no arrow keys); **Space** attack;
  **W/S** attack% ±2%, **D/A** ±0.5%; **B** boats (3.125% tax); **M** auto-attack;
  **P** peace vote; **H** hide UI. **The game has rebindable key actions incl. "Zoom In"/"Zoom Out".**
- Economy: interest tick 0.56s, income tick 5.6s, soft cap 100 troops/px (red),
  hard cap 150/px, 7× boost first 107s, start 12px + 512 balance, land attack tax
  1.17%, **defender 2:1 advantage**.
- Custom Scenario editor: map/player-count/difficulty/colors settings are
  **part canvas-rendered** — the difficulty selector text ("Uniform: Very Easy") is a
  DOM div but clicking it (locator + coords) did NOT open/cycle options (canvas layer
  intercepts). **Vision agent: look at the editor screenshot** (`/home/user/uploads/image.png`
  = the user's setup screen) to figure out the controls.
- **Multiplayer lobby is unreachable from datacenter IPs** (Kaggle etc.) — Custom
  Scenario only (runs client-side offline). That's why the bot plays Custom Scenario.
- Leaderboard is top-left; **your name + color swatch are always there** — OCR it to
  (a) get your color, (b) click it to recenter camera.
- "Choose your start position!" phase = full map visible + stats on screen (that's what
  the 26 screenshots show).

---

## 6. Current data situation

| Dataset | Where | Qty | Status |
|---|---|---|---|
| Real start-phase screenshots | HF `screenshots/` | 26 (23 maps) | ✅ organized + indexed |
| Real maps (validated) | `weights/maps/` + git | 7 | ✅ PASS, 19 rejected |
| Real match recording | HF `recordings/20260807-...` | 123 frames | ⚠️ zoomed-in junk, proves the bug |
| Sim data shards | HF `shard_*.npz` | ~9,443 samples | ✅ workers done |
| Real mid-match screenshots | **NONE yet** | 0 | ⬅️ user may add more |
| Real recordings (good) | **NONE yet** | 0 | ⬅️ THE next milestone |

---

## 7. What to do next (priority order — with vision)

1. **🔴 FIX THE ZOOM** (§0). Use your vision: run the bot, screenshot, SEE the screen,
   find the zoom buttons/keybinds, click your name in the leaderboard to center, then
   zoom out. Verify with the "lit area" / enemy-count success criterion.
2. **🟠 RE-RECORD real matches** once zoom works: `python run_bot.py --record --games 5 --upload`
   (in `/home/user/bot-repo`, `HF_TOKEN=...`). Confirm frames show enemies. Upload to HF
   happens automatically per match.
3. **🟠 Set difficulty to Normal** (user: "on easy bots don't attack each other, put it
   to normal"). The Reset Scenario resets difficulty; find how to click it (see §5) or
   set it before each match. If impossible via automation, accept whatever sticks and
   note it.
4. **🟡 Fine-tune vision on real data** once good recordings exist:
   `scripts/label_real.py --recordings recordings --out weights/nn/real_vision.npz`,
   then `python3 scripts/train_nn.py real` (on Kaggle GPU).
5. **🟡 Kick off Kaggle v8 training** (`kaggle-push/kaggle_train_nn.ipynb` → push to
   Kaggle as `amerameryou/bot-train-nn`, run on GPU). It auto-pulls recordings from HF,
   auto-labels, vision+clone+real, PPO v8 on real maps. Check `kaggle kernels status`
   + pull the log; verify alive-rate/win-rate numbers (loud gates).
6. **🟡 Export best model** to HF: `HF_TOKEN=... python3 scripts/export_hf.py` →
   `amer224/territorial-bot-nn`.
7. **🟢 Update README** (dashboard) with each run's status + session table. Commit+push.
8. **🟢 Long-term**: boats/water actions, more real maps (redo the 19 failed via better
   gray-vs-land discrimination or mid-match screenshots), scale data (~200k varied +
   real, NOT 2M procedural).

---

## 8. Traps & lessons learned (read before touching things)

- **Sandbox resets**: installed packages, playwright chromium, ~/.kaggle, git identity,
  /tmp — all wiped between turns. Reinstall: `pip install -r requirements.txt`,
  `python3 -m playwright install chromium`, re-set git identity, re-write kaggle.json /
  access_token / HF token env. **Files under /home/user persist; /tmp does not.**
- **Workspace budget 128MB / 10k files** — purge `recordings/` after upload, HF cache,
  `weights/nn/dataset.npz` after training. User complained when a 95MB dataset blew it.
- **GitHub README = user dashboard** — keep current, honest numbers, never claim success
  that didn't happen.
- **Silent failures banned**: every stage must print PASS/FAIL and `sys.exit(1)` on fail.
- **P100 (sm_60)**: modern torch refuses it; torch 2.4.1+cu121 works; `_safe_device()`
  tests real ops; FORCE_CPU=1 always available.
- **Playwright headless**: `page.mouse.wheel` and synthetic keys are IGNORED by the
  game's canvas input (trusted-event dispatch needed — see §0 ideas). `page.screenshot()`
  works fine. Buttons found via `getBoundingClientRect` text scan; strict Play matcher.
- **UI overlay**: leaderboard panel is translucent and overlaps the map — the map-extraction
  rect search had to contain the water bbox and snap to the OCR'd aspect ratio.
- **The user plays the game and knows it well** — ask them about mechanics/UI when stuck;
  they've given accurate tips (name-click recenter, difficulty behavior, zoom behavior).
- **User rotates credentials after the project** — never bake tokens into public files;
  keep them in env vars / private notebooks only.

---

## 9. Quick commands cheat-sheet

```bash
# tests
cd /home/user/bot-repo && PYTHONPATH=src python3 -m pytest tests -q        # 47 pass

# run the live bot (record 5 matches, upload)
cd /home/user/bot-repo && HF_TOKEN=<HF_TOKEN> \
  python run_bot.py --record --games 5 --upload --minutes 4

# rebuild maps from screenshots
SHOTS_DIR=/home/user/realdata/shots python3 scripts/rebuild_maps.py

# label real recordings -> vision dataset
python3 scripts/label_real.py --recordings recordings --out weights/nn/real_vision.npz --save-anyway

# train (Kaggle GPU notebook does all of this)
PYTHONPATH=src python3 scripts/train_nn.py vision && ... clone && ... real && ... ppo 60

# export model to HF
HF_TOKEN=<HF_TOKEN> python3 scripts/export_hf.py

# Kaggle kernel status
kaggle kernels status amerameryou/bot-train-nn
kaggle kernels output amerameryou/bot-train-nn -p /tmp/kout

# push to GitHub
git add -A && git commit -m "..." && \
  git push https://<GITHUB_TOKEN>@github.com/amerameryou1-blip/bot.git main
```

---

## 10. Final message to the next agent

You have vision — that's the superpower this project was missing. **Open the game, look
at it, fix the zoom, record real matches, train on real data, and verify with your eyes.**
The user is patient, engaged, and plays the game — use them as a resource. Keep the
README honest, keep failures loud, keep the workspace clean, and push everything to
GitHub + HF. The skeleton is all built and tested; the missing piece is **seeing the map
and getting real training data**. You've got this. 🫡
