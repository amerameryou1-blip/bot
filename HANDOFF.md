# 🤖 HANDOFF — Territorial.io Autonomous Bot (ULTIMATE EDITION)
**Date:** 2026-08-08 · **From:** Arena Agent (no vision) · **To:** successor agent (MUST have vision)
**Goal of this doc:** make the successor fully autonomous — no questions needed. Every link, every command, every fact, every lesson. Copy-paste ready.

---

## 0. TL;DR

A self-playing bot for **territorial.io** (Custom Scenario, offline vs bots) that must win as **LAST SURVIVOR** (elimination, NOT biggest area). Infrastructure is done and committed. The project pivoted to a **continuous sim-based RL loop** (train on thousands of offline simulations against fighting bots). The previous agent stepped aside because the task genuinely needs **VISION** (looking at live frames/screenshots to verify and curate), which it lacks.

**Successor's job:** read this doc → set up the sandbox (Section 7) → run baseline tests → build the continuous sim-RL loop (Section 14) → drive eval win-rate ≥30% vs 12-player mixed/hard → validate in a live match with YOUR eyes → update README dashboard.

---

## 1. Why the previous agent stopped (honest, no sugar)

- The agent has **NO vision**. It cannot look at images/frames — only code-level pixel analysis (colors, blobs, OCR via tesseract).
- The user's core complaint: **recorded real-match data was bad** — ~90% identical green close-up frames of our own territory, because (a) the game spawns zoomed into our spawn and (b) matches ran vs "Uniform: Very Easy" bots that **do not attack each other** → zero combat dynamics in the data.
- User decision (correct): **train on thousands of SIMULATIONS against sim bots that DO fight** — not browser recordings. A vision-capable agent is needed to visually verify live frames, judge map/zoom quality, and validate the final model in real matches.

---

## 2. THE MISSION & SHIP GATES

Win as **LAST SURVIVOR** (by elimination) in territorial.io custom-scenario matches. Official ship gates (from the user):
1. Camera/zoom working: recorded frames show our territory **1–5% of screen** + **≥2 distinct enemy colors** visible → DONE in code (never visually confirmed by eyes — VERIFY!)
2. ≥5 real matches recorded + uploaded to HF → DONE then **DELETED as bad data** (2026-08-08)
3. Vision fine-tuned on real data, per-class gates on held-out real frames: **water ≥97%, me ≥90%, enemy ≥85%, UI ≥98%** → gate implemented & LOUD, never passed (v13 failed it honestly — correct behavior)
4. Policy ≥**30% last-survivor win** vs 12-player mixed/hard sim lobby
5. Real-game proof: bot wins 1 custom-scenario match AS LAST SURVIVOR (by elimination)
6. README updated with honest numbers; model exported to HF; workspace clean

---

## 3. REPO & SOURCE CODE — WHERE EVERYTHING LIVES

### Public repo
- **URL:** https://github.com/amerameryou1-blip/bot
- **Clone (with token, sandbox-safe):**
  ```bash
  git clone https://<GITHUB_TOKEN>@github.com/amerameryou1-blip/bot.git
  ```
- **Clone (tarball if git blocked):**
  ```bash
  curl -L -o bot.tar.gz https://codeload.github.com/amerameryou1-blip/bot/tar.gz/refs/heads/main
  ```
- **Git identity (sandbox resets it every session!):**
  ```bash
  cd bot && git config user.email "amerameryou1-blip@users.noreply.github.com" && git config user.name "amerameryou1-blip"
  ```

### Full repo map (every file, what it does)
```
bot/
├── run_bot.py                 LIVE BROWSER BOT (Playwright headless) — the whole game loop:
│                              fill name → Custom Scenario → strict Play → lobby guard →
│                              detect_own_color (blob diff) → fix_camera (ZOOM FIX) →
│                              ClickLoop (capture→segment→plan→click) → battle_report.json
│                              Flags: --record --games N --minutes M --upload
├── requirements.txt           playwright, pillow, numpy, pytesseract, torch, huggingface_hub, safetensors
├── smoke_imports.py           import sanity
├── README.md                  LIVE DASHBOARD — session table, status, results (user checks this!)
├── HANDOFF.md                 this doc
├── HANDOFF_BEST.md            older best-mode handoff (read it)
├── QUALITY_PLAN.md            quality plan / v8 rewards design (read it)
├── ATTACK_META.md             attack meta analysis (read it)
├── TEST_WEBSITE_PROMPT.md     deep-thinking skill test (ignore)
├── src/
│   ├── bot/
│   │   ├── camera.py          ★ THE ZOOM FIX — synthetic WheelEvents on canvasA at our blob,
│   │   │                         stepped 1.33x search, verify_view() numeric PASS/FAIL gate,
│   │   │                         ZOOM_LEVEL env (auto/0/1/2/3), leaderboard recenter
│   │   ├── ui.py              DOM helpers: _scan_elements (all tags, leaf-preferred),
│   │   │                         find_text (substring, emoji-tolerant), find_play (strict
│   │   │                         emoji-stripped), is_in_lobby (OCR), open_editor,
│   │   │                         enter_custom_match (full nav w/ lobby retry)
│   │   ├── recorder.py        REAL-MATCH RECORDER: frames 256px-wide JPEG (NN input is 64x64),
│   │   │                         clicks.jsonl in ORIGINAL coords + frame_scale in meta,
│   │   │                         meta.json (self/enemy colors, zoom, cam gate), HF upload
│   │   ├── planner.py         ClickPlanner: expand cheap → attack weakest/drained neighbor →
│   │   │                         bank to red. Config now AGGRESSIVE: attack_density=42 (was 75),
│   │   │                         attack_balance_ratio=1.5 (was 2.0), attack_pct=15, ratio_max=0.75
│   │   ├── calibration.py     saturated_colors, blob, edges_touched, dominant_colors,
│   │   │                         swatch_from_strip, _ocr_words (tesseract), find_name_box,
│   │   │                         calibrate_from_leaderboard, center_lock_calibrate, ocr_map_info
│   │   ├── click_loop.py      decision loop (capture → segment → plan → click at cadence)
│   │   ├── vision.py          segment() color-based scene parse (me/enemy/neutral/water)
│   │   ├── config.py          Palette, PlayerColor, LoopConfig
│   │   ├── economy.py         TroopTracker (balance/land/income/interest/caps)
│   │   └── controls.py        MouseControls (real clicks via Playwright)
│   ├── sim/
│   │   ├── game6.py           ★ PRIMARY SIM (v8): loads REAL maps from weights/maps/*.npz
│   │   │                         (int8: -2 mountain impassable, -1 water, 0 land), 8-15 bots,
│   │   │                         mixed skill lobbies, vivid per-player colors, kill tracking,
│   │   │                         last-survivor win, same interface as game5
│   │   └── game5.py           older procedural sim (lakes/island/mountains/desert/swamp, 3-4 bots)
│   └── nn/
│       ├── model.py           ★ TerritoryNet — 85,533 params, CPU-fast ~5-10Hz.
│       │                         input 64x64 RGB → 4 conv blocks → 16x16 feature map →
│       │                         seg head (5 classes: 0 water 1 neutral 2 me 3 enemy 4 ui)
│       │                         + localize head (my centroid) + click-map head (16x16 logits =
│       │                         WHERE TO CLICK — the per-pixel deliverable) + kind head
│       │                         (expand/attack/bank) + pct head (sigmoid) + value head (RL)
│       ├── data.py             sim data collection (collect_parallel / collect_single)
│       └── bot_brain.py        inference wrapper
├── scripts/
│   ├── train_nn.py            ★ THE TRAINER — stages: collect → vision → clone → real → ppo → eval
│   │                             (run: python3 scripts/train_nn.py <stage> [extra])
│   │                             - v8 rewards: growth/2000 + kill x2 + win +5 (last survivor!)
│   │                               - idle penalty, tiny survival tick 0.005
│   │                             - PPO curriculum: easy→medium→hard AND enemies 2→…→10 by alive-rate
│   │                             - FIXED: global advantage norm, reward x10, save best-eval model
│   │                             - real stage: capped sqrt-inverse weights, cosine LR, sim-mixing,
│   │                               rare-frame oversampling, LOUD gates (raises on fail)
│   ├── label_real.py          REAL-FRAME AUTO-LABELER: 5-class per-pixel + click targets (16x16 cell
│   │                             + kind + pct) from recordings; uint8 output (OOM fix);
│   │                             clicks normalized via meta frame_orig_size
│   ├── record_batch.py        sequential real-match recorder (crash-safe, per-match timeout,
│   │                             zoom rotation 2/3/2/1/2/3 for more 'me' pixels, session summary)
│   ├── push_github.py         HF upload pipeline: zip NEW sessions → private GH repo →
│   │                             migrate kernel (token-injected temp copy) → HF → DELETE local
│   ├── launch_train.py        ★ GPU LAUNCHER: injects HF_TOKEN into TEMP copy of the notebook,
│   │                             converts to SCRIPT kernel (notebooks don't auto-execute via API!),
│   │                             pushes, monitors, dumps log. Usage:
│   │                             HF_TOKEN=... KAGGLE_CONFIG_DIR=... python3 scripts/launch_train.py
│   ├── export_hf.py           model.pt → safetensors + config + model card → amer224/territorial-bot-nn
│   ├── merge_worker_data.py   pull + merge sim shard_*.npz from HF into dataset.npz
│   ├── collect_worker.py      Kaggle CPU worker entry: sim collect shards → HF (env-driven)
│   ├── rebuild_maps.py        extract coastline/water/mountain masks from the 26 start-phase
│   │                             screenshots; validates % vs OCR within ±5 → loud PASS/FAIL.
│   │                             7 maps PASS: island, desert, pond, island_kingdom, middle_east,
│   │                             white_arena, black_arena. 19 FAIL (ambiguous gray) — rejected loudly
│   ├── upload_recordings.py   HF upload helper (old)
│   ├── validate_meta.py / validate_weights.py / check_vision.py / train_weights.py  misc tools
├── kaggle-push/
│   ├── kaggle_train_nn.ipynb  GPU trainer notebook (TOKEN-FREE; launcher injects at push)
│   └── migrate_kernel/        HF migration kernel (TOKEN-FREE; env injected at push)
├── tests/                     47 tests, all passing (baseline gate)
├── weights/
│   ├── maps/*.npz             ★ 7 VALIDATED REAL MAPS (committed to git) — island, desert, pond,
│   │                             island_kingdom, middle_east, white_arena, black_arena
│   └── nn/                    gitignored — dataset.npz, real_vision.npz, model.pt (deleted, clean)
├── .gitignore                 recordings/, weights/nn/, bot_output/, probe_*, bot-train-nn.ipynb
└── HANDOFF files              see above
```

---

## 4. ALL LINKS (every URL, every service)

### The game
| What | URL |
|---|---|
| Game | https://territorial.io/ |
| Game source (657KB inline JS — zoom internals live here) | View-source of https://territorial.io/ |

### GitHub
| What | URL |
|---|---|
| Public repo (THE repo) | https://github.com/amerameryou1-blip/bot |
| Repo tarball (if git blocked) | https://codeload.github.com/amerameryou1-blip/bot/tar.gz/refs/heads/main |
| Private transfer repo **DELETED 2026-08-08** (data ferry) | https://github.com/amerameryou1-blip/bot-recordings (gone) |
| GitHub API | https://api.github.com |

### Kaggle kernels (all under https://www.kaggle.com/code/amerameryou/)
| Kernel | Purpose | Status |
|---|---|---|
| `bot-train-nn` | GPU trainer (P100) — continuous target | v14 may STILL BE RUNNING — check/cancel! |
| `bot-train` | older trainer | old |
| `worker-island` / `worker-mountains` / `worker-desert` / `worker-swamp` | CPU sim-data collectors (4 workers) | complete |
| `migrate-recordings-b1/b2/b3` | recordings → HF ferry (via GitHub or dataset attach) | complete |
| `migrate-labels-b1` | labeled npz → HF | complete |
| `check-hf-data` | verify HF dataset contents (session listing) | complete |
| `delete-bad-data` | deleted recordings/ + realdata/ + model repo from HF | complete |
| `hf-access-test` | proved HF works from Kaggle network (4/4 PASS) | complete |
| `notebook*` (10e5cd8702, 7fc15f4537, 7c3c5f9b2d, 165c30e79f, 1693db30e6, 6de2c52e48, 267ce1802b, c0ef73ee11) | old user notebooks | ignore |

### Kaggle datasets (all under https://www.kaggle.com/datasets/amerameryou/)
| Dataset | Purpose | Status |
|---|---|---|
| `territorial-bot-recordings-b1/b2/b3` | recording zips (bad data) | private, could NOT be API-deleted — delete in web UI if desired |
| `territorial-bot-labels-b1` | labeled npz | same — delete in web UI |
| `the-game`, `github-repo`, `thegame` | user's old datasets | ignore |

### Hugging Face (all under https://huggingface.co/)
| Repo | Purpose | Status |
|---|---|---|
| `datasets/amer224/territorial-bot-data` | MAIN DATA REPO (private) — screenshots/ + shard_*.npz kept; recordings/ + realdata/ DELETED | live |
| `amer224/territorial-bot-nn` | model repo | **DELETED 2026-08-08** |

### Data (kept — the good stuff)
- `screenshots/<map>/*.png` — 26 REAL start-phase map screenshots (all maps incl. africa, asia, australia, black_arena, british_isles, caucasia, cliffs, desert, europe, halo, island, island_kingdom, mare_nostrum, middle_east, mountains, north_america, pond, …) + `index.json`
- `shard_<map>_medium_s*.npz` — sim data from old workers (island/mountains/desert/swamp/lakes; ~12,798 samples merged)

### Tools/docs
| What | URL |
|---|---|
| PyTorch P100 build (sm_60; the ONLY modern torch that works on P100) | https://download.pytorch.org/whl/cu121 (`pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`) |
| Kaggle API docs | https://www.kaggle.com/docs/api |
| Hugging Face hub docs | https://huggingface.co/docs/huggingface_hub |
| Playwright | https://playwright.dev/python/ |

---

## 5. CREDENTIALS — ⚠️ ALL PLACEHOLDERS (real tokens live ONLY in the user's chat history)

| Service | Account | Placeholder | Where used |
|---|---|---|---|
| GitHub | `amerameryou1-blip` | `ghp_***_REPLACE_ME` | git clone/push (token-in-URL), GitHub API |
| Kaggle | `amerameryou` | `KGAT_***_REPLACE_ME` | `~/.kaggle/kaggle.json` = `{"username":"amerameryou","key":"<KEY>"}` + `~/.kaggle/access_token` = `<KEY>` ; also `KAGGLE_CONFIG_DIR=/tmp/kagglecfg` + `KAGGLE_API_TOKEN` env pattern |
| Hugging Face | `amer224` | `hf_***_REPLACE_ME` | env `HF_TOKEN` (used by kernels / export) |

**THE USER'S RULE:** credentials rotate when the project ends (now). Real tokens were pasted in chat history only — **never** commit them (GitHub secret-scan blocks token commits; it caught leaks twice). Inject at push time (patterns: `scripts/launch_train.py`, `scripts/push_github.py`).

---

## 6. FRESH SANDBOX SETUP — copy-paste, run top to bottom

> Sandbox resets EVERY session: pip packages, playwright chromium, system libs, ~/.kaggle, git identity, /tmp all wiped. Workspace files under /home/user persist. Do this every time.

```bash
# 1) clone
cd /home/user
git clone https://<GITHUB_TOKEN>@github.com/amerameryou1-blip/bot.git
cd bot
git config user.email "amerameryou1-blip@users.noreply.github.com"
git config user.name "amerameryou1-blip"

# 2) deps
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch for local tests
pip install playwright kaggle pytesseract
python3 -m playwright install chromium

# 3) system libs for headless chromium (Ubuntu; needed after every reset)
sudo apt-get update -qq
sudo apt-get install -y -qq libnspr4 libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 \
  libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2t64 \
  libpango-1.0-0 libcairo2 libx11-6 libxcb1 libxext6 libxi6 libxtst6 libglib2.0-0 libfontconfig1 \
  libfreetype6 libexpat1
sudo apt-get install -y -qq tesseract-ocr

# 4) kaggle creds
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
echo '{"username":"amerameryou","key":"<KAGGLE_KEY>"}' > ~/.kaggle/kaggle.json
echo -n '<KAGGLE_KEY>' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/kaggle.json ~/.kaggle/access_token
export KAGGLE_CONFIG_DIR=~/.kaggle
export KAGGLE_API_TOKEN=<KAGGLE_KEY>

# 5) verify
kaggle kernels list --user amerameryou --page-size 5
python3 -c "import torch, playwright, pytesseract; print('deps OK')"

# 6) BASELINE GATE (must pass!)
PYTHONPATH=src python3 -m pytest tests -q     # expect 47 passed, 0 failed
```

---

## 7. COMMANDS CHEAT SHEET (everything used, grouped)

### Live bot
```bash
# play 1 match, 3 minutes, record frames+clicks
BOT_NAME="AureliaBot" PLAY_MINUTES=3 python3 run_bot.py --record --games 1 --minutes 3

# play N consecutive matches
python3 run_bot.py --record --games 5 --minutes 4

# batch recorder (crash-safe, zoom rotation, summary table)
python3 scripts/record_batch.py --games 6 --minutes 2 --tag island-b7

# env vars the bot reads
BOT_NAME, PLAY_MINUTES, DECISION_HZ, ZOOM_LEVEL (auto/0/1/2/3), MANUAL_COLOR (JSON rgb),
REC_FRAME_WIDTH (default 256), HF_TOKEN (for --upload)
```

### Labeling
```bash
python3 scripts/label_real.py --recordings recordings --out weights/nn/real_vision.npz --save-anyway
python3 scripts/label_real.py --images <folder> --out ... --save-anyway
```

### Training (local CPU smoke / Kaggle GPU)
```bash
# local: single stage (collect/vision/clone/real/ppo/eval)
FORCE_CPU=1 COLLECT_SEEDS=6 WORKERS=2 python3 scripts/train_nn.py collect
FORCE_CPU=1 python3 scripts/train_nn.py real

# Kaggle GPU (the real deal) — pushes a SCRIPT kernel, monitors, dumps log
HF_TOKEN=<HF> KAGGLE_CONFIG_DIR=~/.kaggle KAGGLE_API_TOKEN=<KG> python3 scripts/launch_train.py
# env: COLLECT_SEEDS (default 60), WORKERS (4), PPO_ROUNDS (100), SIM_BOTS, SIM_TICKS, SIM_MAP, FORCE_CPU
```

### Kaggle CLI
```bash
kaggle kernels list --user amerameryou --page-size 20
kaggle kernels status amerameryou/bot-train-nn            # COMPLETE/ERROR/RUNNING/QUEUED
kaggle kernels pull amerameryou/bot-train-nn              # fetch source
kaggle kernels push -p /tmp/kerneldir                     # push script/notebook kernel
kaggle kernels output amerameryou/bot-train-nn -p /tmp/out   # download output (log FIRST, then working dir — use --force; it's big)
kaggle datasets create -p /tmp/dsdir -q
kaggle datasets status amerameryou/<dataset>
kaggle datasets version -p /tmp/dsdir -m "msg"            # bump version (needed before kernels can attach!)
```

### HF (via kernels only from sandbox — direct is 429-blocked!)
```python
# inside a Kaggle kernel (enable_internet=true, is_private=true):
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN",""))
api.upload_folder(folder_path=sess, path_in_repo=f"recordings/{sid}",
                  repo_id="amer224/territorial-bot-data", repo_type="dataset",
                  commit_message="...")
api.delete_folder(repo_id=..., repo_type="dataset", path_in_repo="recordings", ...)
api.list_repo_files("amer224/territorial-bot-data", repo_type="dataset", token=...)
api.upload_file(path_or_fileobj=..., path_in_repo=..., ...)
```

### Git
```bash
git add -A && git commit -m "msg"
git push https://<TOKEN>@github.com/amerameryou1-blip/bot.git main
# if rejected: git pull --rebase ... then push (conflict markers <<<<<<< = fix by keeping your side)
# secret-scan rejection = a token leaked into the tree → find it, remove, reset --soft to last clean commit, recommit
```

### Transfer pipeline (recordings → HF when sandbox HF is blocked)
```bash
HF_TOKEN=<HF> GH_TOKEN=<GH> KAGGLE_CONFIG_DIR=~/.kaggle KAGGLE_API_TOKEN=<KG> \
  python3 scripts/push_github.py            # zip new sessions → private GH repo → migrate kernel → HF → deletes local
```

---

## 8. GAME KNOWLEDGE — everything the user taught us + everything we reverse-engineered

### Controls (user-verified)
- **Double-click = claim land** (canvas-only; no arrow keys)
- **Space = attack** · **W/S = attack% ±2%** · **D/A = ±0.5%**
- **B = boat** (3.125% tax) · **M = auto-attack** · **P = peace vote** · **H = hide UI**
- Rebindable key actions exist incl. **"Zoom In" / "Zoom Out"** (in the key-action table near "Switch UI Visibility")

### Economy (user-verified facts)
- Interest tick **0.56s**, income tick **5.6s**
- Soft cap **100 troops/px** (red), hard cap **150/px**
- **7× boost first 107s**
- Start = **12px + 512 balance**
- Land attack tax **1.17%**, defender **2:1 advantage**
- Boats 3.125% tax

### Custom Scenario editor (user + verified)
- Editor rows: Map · Player Count · Game Mode (Battle Royale/Teams) · Colors (Random) · Bot Difficulty (Uniform: Very Easy) · Spawning (Random) · Player Names · Additional Income · Territorial Income · Interest Income · Starting Resources · More Settings (Reset Scenario / Open File / Save As)
- **"Bot Difficulty → Uniform: Very Easy": DOM `<div>` row but clicks DO NOT cycle it** (canvas intercepts) — could not automate → matches run Very Easy (bots don't fight each other — THE data problem)
- **"Reset Scenario" EXITS the editor back to the main menu** (empirically verified) — never click it; editor opens with correct defaults (Battle Royale, Colors Random, Very Easy, Spawning Random)
- **"Map" row → "Settings"** opens the map-type panel: Procedural Map / Realistic Map / Custom Map (DOM `<p>` rows — clicking moves the radio). Map-name list is canvas-rendered (flaky headless). Procedural maps: White Arena, Black Arena, **Island (default)**, Mountains 1, Desert, Swamp, White Plains, Cliffs, Pond, Halo, Island Kingdom, Mountains 2. Realistic maps: Europe, World 1, Caucasia, Africa, Middle East, Scandinavia, North America, South America, Asia, Australia, World 2, British Isles, Mare Nostrum.
- "Choose your start position!" phase = full map + dimension/land/water/mountain % stats visible (source of the 26 screenshots)

### UI layout facts (for OCR/vision)
- Leaderboard **top-left**; your name + color swatch ALWAYS visible (OCR it: `_ocr_words`, `calibrate_from_leaderboard`)
- Bottom bar (troop slider), top banner (player count), left edge = UI zones
- Leaderboard panel is **translucent** — map shows through (masking the whole rect pollutes labels; use bright-text rule instead)

### Game JS internals (reverse-engineered from the 657KB inline script)
- Zoom core (private, NOT on window): `du.zoom(ft,fg,fh){hs*=ft; hq=(hq+fg)*ft-fg; hr=(hr+fh)*ft-fh;}`
- Wheel path: `canvasA.addEventListener('wheel', fb)` (`S[73]='wheel'`) → `fb(eL)` → `xO` (preventDefault) → `eZ.fb(clientX,clientY,deltaY)` → `fi=500/(500+deltaY)` clamped `[0.5,2]` → `du.zoom(fi,...)+fk(fi,...)`; zoom centered on CURSOR
- Clamp: `fj(ft){ft*ay>1024→1024/ay; ft*ay<0.125→0.125/ay}`
- Camera reset `iJ()` (hs=1,hq=hr=0) only on resize/menu re-render
- `eZ.eT()` blocks wheel DURING zoom animations → space synthetic ticks ~0.7s apart
- `fq.fr()` = viewport bounds recompute; camera follow only in the zoom-animation path (no per-frame auto-follow)
- **No `isTrusted` check** on the wheel handler → synthetic `new WheelEvent('wheel',{deltaY,clientX,clientY,bubbles:true,cancelable:true})` dispatched via `page.evaluate` WORKS
- `du` is a local `new hb()` in the IIFE — NOT reachable from window (can't call du.zoom directly)
- Editor "Map Settings" panel is DOM `<p>` rows (findable via `_scan_elements`); difficulty row is DOM but its click handler is canvas-blocked

### User's gameplay tips (they play this game — trust them)
1. **"When you choose your spawn spot it zooms in to your location... zoom out a reasonable amount so the bot can see other bots"** — the zoom blocker
2. **"Fetch your name from the leaderboard (always visible), click it → camera recenters on you, then zoom out"** — recenter trick (implemented: `recenter_via_leaderboard`)
3. **"On easy bots don't attack each other"** — easy gives no combat data → difficulty matters (couldn't automate; sim bots are the workaround)
4. **Reset Scenario resets difficulty** and the editor control is partly canvas-rendered

---

## 9. EVERYTHING DONE & VERIFIED (with evidence)

### ✅ THE ZOOM BLOCKER — FIXED (code-verified; EYES-STILL-NEEDED)
- Found zoom core + wheel path (Section 8). Fix in `src/bot/camera.py`: synthetic wheel events on `canvasA` AT OUR BLOB's screen position, stepped 1.33× search until blob = 1–6% screen AND ≥2 distinct enemy clusters (dedupe + UI/self/dominant-color filters), numeric `verify_view()` PASS/FAIL, `ZOOM_LEVEL` env, forced-zoom retry + clamp fallback to auto search.
- Evidence: 31 recorded matches, ~80% camera-gate PASS (self 0.3–0.5% → 1.2–1.9%, enemies 2–6). Examples: `[camera] PASS after 1 tick(s): self=1.7%, enemies=3`.
- ⚠️ **Verify with your eyes:** the numbers were computed, never seen. Look at `bot_output/cam_ok_*.png` or a recording frame and confirm the view is genuinely good.

### ✅ MENU/LOBBY BUGS FIXED
- Menu items are `<div>/<p>` with emoji prefixes (`🗡️\nCustom Scenario`); button-only scans missed them → bot sometimes clicked main-menu Play → **multiplayer lobby** (not a match). Fixed: all-element scan (leaf-preferred), strict emoji-stripped "Play", `is_in_lobby()` OCR guard + auto-retry. `[nav] confirmed: in a custom scenario match (not the lobby)`.

### ✅ TRAINING BUGS FOUND & FIXED (the crown jewels)
1. **Labeler mislabeled dark map pixels as UI** (dark navy ocean, dark land failed brightness floors) → NN could never learn water. FIX: UI = known screen regions + bright low-chroma text only; no darkness floor; neutral fallback; territory overrides water. Result: water purity 98.5%, enemy class 2.5%→11%, UI 49.8%→14.8%.
2. **SILENT GATE BUG:** `classify_acc()` returns INT keys (0–4) but the gate checked STRING keys (`acc.get("water")` always None) → gate ALWAYS "PASSED". v5 shipped a collapsed model "successfully". FIX: int keys + user gates (water≥97, me≥90, enemy≥85, ui≥98) + `raise RuntimeError` on failure. **v13 proved it: run failed LOUDLY at 3403s instead of shipping garbage.**
3. **uint8→float input bug:** real stage fed model 0–255 frames while sim pretrained on 0–1 → transfer wrecked. FIX (v14): `rgb_all = d["rgb"].astype(np.float32)/255`.
4. **PPO loss=0.000 bug:** per-episode advantage normalization zeroes gradients on near-constant rewards. FIX: global batch normalization + reward×10 + entropy 0.05 + save BEST eval model (not last round). Verified: loss 0.346 on the exact case that gave 0.000.

### ✅ GPU RUN HISTORY (Kaggle `bot-train-nn`, P100)
| Run | What happened |
|---|---|
| v5 | Ran end-to-end → exported model to HF. BUT real stage collapsed to all-UI and the gate "passed" (bug #2). **Not trusted.** |
| v13 | Fixed labeler pulled 26 sessions / 5,635 real frames; vision hit **win-rate 0.33 / avg_rank 2.83 (best ever)**; real stage failed the gate → **stopped loudly at 3403s** (CORRECT behavior). |
| v14 | Launched 2026-08-08 with ALL fixes (uint8 fix, capped sqrt-inverse weights [0.2,0.8], cosine LR 3e-5, sim-mixing every 2nd batch, rare-frame oversampling, 12 epochs). **MAY STILL BE RUNNING — check `kaggle kernels status amerameryou/bot-train-nn` and cancel if done.** |

### ✅ TESTS
`PYTHONPATH=src python3 -m pytest tests -q` → **47 passed, 0 failed** (was 43+1 in the original handoff; 47 in the last verified run).

---

## 10. WHAT WAS DELETED (clean slate — nothing lost that matters)
- Local: `recordings/` (31 sessions ~6,500 frames), `weights/nn/*` (npz/models), probe artifacts
- HF `amer224/territorial-bot-data`: `recordings/` + `realdata/` (7,338 files) — deleted via kernel; **KEPT**: `screenshots/` (26 maps) + `shard_*.npz` (sim data)
- HF model repo `amer224/territorial-bot-nn` — deleted
- GitHub private transfer repo `bot-recordings` — deleted
- Kaggle datasets `territorial-bot-recordings-b1/b2/b3`, `territorial-bot-labels-b1` — private; NO API delete exists; delete in web UI if you want them gone (harmless)

---

## 11. DATA INVENTORY (Hugging Face `amer224/territorial-bot-data`, private)
- ✅ `screenshots/<map>/*.png` — 26 real start-phase map screenshots + `index.json` (GOLD for map extraction / vision)
- ✅ `shard_<map>_medium_s*.npz` — sim data from the 4 old CPU workers (~12,798 samples merged: island 2,235 · mountains 2,445 · desert 2,356 · swamp 2,407 · lakes ~300)
- ❌ `recordings/`, `realdata/` — deleted (bad data)

---

## 12. TRAPS & LESSONS (everything learned, hard-won)
1. **Sandbox resets every session** — reinstall deps/chromium/libs/kaggle/git-identity each time (Section 6).
2. **Workspace budget 128MB / 10k files** — push data out, delete locally; `.cache` (chromium ~775MB) is excluded from snapshots but everything else counts.
3. **HF blocks this sandbox's IP** (HTTP 429 on ALL endpoints incl. web UI) — route HF ops through Kaggle kernels (works from Kaggle's network; proven 4/4 PASS).
4. **Kaggle notebook kernels pushed via API do NOT auto-execute** — use SCRIPT kernels (`kernel_type: "script"`). The v5-v12 "COMPLETE with empty log" mystery was exactly this.
5. **Kaggle dataset attach is flaky:** fresh datasets fail with "New Datasets cannot be attached in non-interactive sessions" — create → `datasets version` (bump) → WAIT ~1-2 min → attach. Kagglehub download path also flaky for fresh datasets. The reliable pattern: attach via `dataset_sources` AFTER a stable version, then glob `/kaggle/input/<slug>/**/meta.json` (Kaggle auto-extracts zips into session dirs!). Or: private GitHub repo → kernel clones it.
6. **No API delete for Kaggle datasets/kernels** — web UI only.
7. **GitHub secret-scan blocks token commits** (caught 2 leaks: a hardcoded HF token in a script, a stray pulled notebook) — keep repo token-free; inject at push time; `git reset --soft` to last clean commit to purge history.
8. **P100 (sm_60):** modern torch refuses it — auto-install `torch==2.4.1+cu121` (has sm_60 kernels); `cuda_ok()` test; `FORCE_CPU=1` fallback. GPU kernels cap ~9–12h → loops must resume (watchdog).
9. **Kaggle GPU budget ~36h/week, 5 concurrent CPU kernels max** — no "70 tabs"; one GPU kernel at a time.
10. **Playwright headless:** `page.mouse.wheel` and synthetic keyboard are IGNORED by the game canvas; only `page.screenshot()` + dispatchEvent-based synthetic events (wheel) work. Buttons via `getBoundingClientRect` text scan.
11. **The leaderboard is translucent** and overlaps the map — map-extraction rect search had to contain the water bbox and snap to OCR'd aspect ratio.
12. **OOM:** the sandbox has ~1.9GB RAM — label 5,500+ frames in float32 lists → killed (exit 137). Keep arrays uint8 (4× smaller) and use mmap for big npz.
13. **Battle reports are crash-safe** (incremental JSON) and recorders flush per line — never lose data on crash.
14. **Multiplayer lobby is UNREACHABLE from datacenter IPs** — Custom Scenario only, always.

---

## 13. USER PREFERENCES (stated directly — honor EVERY one)
1. **QUALITY OVER QUANTITY, ALWAYS.** Highest quality first. Real screenshots/recordings/clicks/maps/colors. Sim data = filler AFTER real data exists. NEVER run a 2M-sample procedural farm (explicitly rejected).
2. **REAL-WORLD FIDELITY:** multiple enemies (8–15), DIFFERENT vivid colors every match (Colors: Random), real water/land, real maps.
3. **THE BOT MUST KNOW EXACTLY WHERE TO CLICK, PER PIXEL** — the click-head (16×16 click-map) is a core deliverable, trained on REAL recorded clicks.
4. **WIN = LAST SURVIVOR (by elimination)** — not biggest area. Reward win +5, kill bonus, growth, idle penalty (v8 — keep).
5. **FAIL LOUDLY** — every stage prints concrete PASS/FAIL numbers, exits nonzero. A silent "COMPLETE" that didn't learn = hard fail. (User burned twice.)
6. **THINK DEEP, DON'T RUSH** — analyze → plan → verify → implement → re-verify → document, per sub-task. Write reasoning down.
7. **USE VISION CONSTANTLY** — look at the live game, screenshots, recorded frames, the editor. (THIS is why a vision agent was requested.)
8. **WORKSPACE HYGIENE:** 128MB/10k files cap — upload to HF, delete locally. Keep repo clean.
9. **KEEP PRODUCING DATA** — parallel processes, keep recording; never let a crash end the stream; push to HF first, delete locally.
10. **FULLY AUTONOMOUS** — no babysitting, no manual zoom. Zoom automated. No human-in-the-loop unless truly stuck.
11. **DIFFICULTY = NORMAL when possible** — couldn't be automated (canvas intercepts); documented honestly. Sim bots at mixed skill compensate.
12. **2–4 MATCHES PER MAP, SWITCH MAPS BETWEEN BATCHES** — depth on a map beats thin coverage. Default map Island is fine.
13. **README = THE DASHBOARD** — honest run status + session table, always updated.
14. **CREDENTIALS ROTATE WHEN THE PROJECT ENDS** — never hardcode; env vars only. (Project ended → rotate now.)
15. **TRAINING COMPUTE:** ~36 GPU-hrs/week on Kaggle, 5 concurrent CPU sessions. Real-data fine-tunes and targeted PPO beat giant procedural runs. Models ship to HF.

---

## 14. THE NEW DIRECTION — CONTINUOUS SIM RL LOOP (build this)

**Why:** browser recordings gave bad data (zoom + passive Very Easy bots). The sim (`game6.py`) has REAL maps, 8–15 bots, mixed skill, bots that DO fight, real economy/combat rules, v8 rewards, curriculum — and generates its own (state, action, reward) data endlessly. This is the standard "runs for hours and learns on its own" recipe (AlphaZero-style: play → collect → update → play again with the better policy).

```
        ┌────────────────────────────────────────────────────────────┐
        │  CONTINUOUS RL LOOP (never stops between sessions)         │
        │                                                            │
  ┌─────▼──────┐   trajectory shards (HF)   ┌───────────────────┐    │
  │ 5× Kaggle  │ ─────────────────────────▶ │ 1× Kaggle GPU     │    │
  │ CPU workers│   (states/actions/rewards) │ PPO trainer loop   │    │
  │ play sim   │                            │ pull shards → train│    │
  │ matches    │ ◀───────────────────────── │ → save checkpoint  │    │
  └────────────┘   latest checkpoint (HF)   │ → eval stats (HF)  │    │
                                           └───────────────────┘    │
  Watchdog: re-push kernels when sessions end (9-12h caps) → loop    │
  effectively never stops.                                           │
  └──────────────────────────────────────────────────────────────────┘
```

**v1 (build now — everything else already exists):**
1. Worker loop mode: `collect_worker.py` plays matches with the LATEST checkpoint + ε-exploration, pushes shards → HF, pulls newer checkpoint, repeats until session cap.
2. Trainer loop mode: pull new shards → `stage_ppo` (fixed) → save checkpoint + eval (win-rate/rank) → HF, repeat.
3. Watchdog script: `kaggle kernels push` when sessions end; README updates with eval trend.

**v2 (self-play — the "gets better forever" mechanism):**
4. Add policy-controlled opponents to `game6.py`: some bots load OLDER checkpoints from a pool → trains against a rising bar.

**Eval gate:** ≥30% win vs 12-player mixed/hard before live validation.
**Live validation (needs YOUR eyes):** `python3 run_bot.py --record --games 3 --minutes 3` → look at frames → confirm zoom/view/enemies → get a last-survivor win.

---

## 15. FAQ — questions you might ask, answered (where the answer lives)

**Q: What am I building?** A: A bot that plays territorial.io custom scenarios and wins as LAST SURVIVOR. Mission + gates: Section 2.

**Q: Where is the source code?** A: https://github.com/amerameryou1-blip/bot — clone per Section 3; full file map in Section 3; everything committed on `main` (last commit `65efe73`).

**Q: Where is the zoom fix?** A: `src/bot/camera.py` (synthetic wheel events + stepped search + numeric gate). JS internals: Section 8. Evidence: Section 9.

**Q: Why did the previous agent stop?** A: No vision (Section 1). The task needs eyes; a vision-capable agent was requested.

**Q: Is the data any good?** A: The old real recordings were BAD and DELETED (Section 10). KEPT: 26 map screenshots + sim shards (Section 11). The new plan generates its own good data via sim RL (Section 14).

**Q: What's the neural net?** A: TerritoryNet, 85,533 params — seg (5-class) + localize + 16×16 click-map + kind + pct + value. `src/nn/model.py`.

**Q: How do I record a real match?** A: `python3 run_bot.py --record --games N --minutes M` or `scripts/record_batch.py`. Commands: Section 7.

**Q: How do I get recordings to HF?** A: `scripts/push_github.py` (private GH → migration kernel → HF). Direct HF from sandbox is 429-blocked. Section 7.

**Q: How do I train on the GPU?** A: `HF_TOKEN=... python3 scripts/launch_train.py` (pushes script kernel, monitors, dumps log). Env knobs: Section 7.

**Q: How do I label real frames?** A: `scripts/label_real.py --recordings recordings --out weights/nn/real_vision.npz --save-anyway`.

**Q: Where are the real maps?** A: `weights/maps/*.npz` (7 validated) + `scripts/rebuild_maps.py` + the 26 screenshots on HF for extracting more.

**Q: What's the win/reward structure?** A: v8 — growth/2000 + kill×2 + WIN +5 (last survivor) − idle penalty − tiny survival tick. In `scripts/train_nn.py` (`_rollout_one`).

**Q: How do the Kaggle workers work?** A: `worker-island/mountains/desert/swamp` kernels → `scripts/collect_worker.py` → shard_*.npz → HF; merge via `scripts/merge_worker_data.py`. Pull them: `kaggle kernels pull amerameryou/worker-island`.

**Q: What were the run results?** A: v5 (shipped broken, silent gate bug), v13 (failed loudly, vision 0.33 win-rate best-ever), v14 (launched with all fixes, check status). Section 9.

**Q: What GPU/torch?** A: P100 sm_60 → torch 2.4.1+cu121 only (auto-handled). ~36h/week budget. Trap #8.

**Q: What are the tokens?** A: Placeholders in this doc; real ones in chat history; ROTATE THEM (project ended). Section 5.

**Q: Is the multiplayer reachable?** A: NO from datacenter IPs — Custom Scenario (offline bots) only. Trap #14.

**Q: How do I check the GPU kernel status?** A: `kaggle kernels status amerameryou/bot-train-nn` (v14 may still be running — cancel in web UI if done).

**Q: What does "done" look like?** A: Section 2 ship gates — incl. ≥30% sim win-rate and a real last-survivor win, verified with eyes.

---

## 16. FIRST STEPS FOR THE SUCCESSOR (do in this order)
1. **Check/cancel the v14 GPU kernel** (burning budget if still running): `kaggle kernels status amerameryou/bot-train-nn`.
2. Read `QUALITY_PLAN.md`, `HANDOFF_BEST.md`, `ATTACK_META.md` in the repo.
3. Full sandbox setup (Section 6) — expect **47 tests passing**.
4. **Build the continuous sim RL loop (Section 14 v1):** worker loop-mode + trainer loop-mode + watchdog.
5. Drive eval win-rate vs 12-player mixed/hard lobby ≥30% (loud gates each round).
6. **USE YOUR VISION:** validate in live matches — run `python3 run_bot.py --record --games 3 --minutes 3`, LOOK at frames, confirm zoom/view/enemies, get the last-survivor win.
7. Update README (dashboard) with honest numbers; export model to HF; keep workspace clean.

---

*End of ultimate handoff. The repo is clean, honest, and ready. The bot is one good vision module + one continuous RL loop away from winning.*
