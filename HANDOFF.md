# 🤖 HANDOFF — Territorial.io Autonomous Bot
**Date:** 2026-08-08 · **From:** Arena Agent (no vision) · **To:** successor agent (MUST have vision)

---

## 1. TL;DR

A self-playing bot for **territorial.io** (Custom Scenario, offline vs bots) that must win as **LAST SURVIVOR** (elimination, not biggest area). Massive progress was made on infrastructure, the zoom blocker, and data plumbing — then the project pivoted to a **sim-based continuous RL loop** (the right architecture) and the previous agent stepped aside because the task genuinely needs **vision** (looking at live game frames / screenshots to verify and curate data), which it does not have.

**The successor's job:** take the clean repo, build/run the continuous sim RL loop (Section 7), and verify the final model in live matches — with YOUR eyes.

---

## 2. Why the previous agent stopped (honest)

- The agent **has NO vision** — it cannot look at images/frames; it can only do code-level pixel analysis (colors, blobs, OCR).
- The user's core complaint: **the recorded real-match data was bad** — ~90% identical green close-up frames of our own territory, because (a) the game spawns zoomed into our spawn and (b) matches ran vs "Uniform: Very Easy" bots that **do not attack each other** → no combat dynamics in the data.
- The user (correctly) concluded the path forward is: **train on thousands of offline SIMULATIONS against fighting bots**, not on browser recordings. A vision-capable agent is needed to (1) visually verify live-game frames, (2) judge map/zoom quality, (3) validate the final model in a real match.

---

## 3. The mission (unchanged)

Win as **LAST SURVIVOR** in territorial.io custom-scenario matches. Ship gates:
1. Camera/zoom working (recorded frames show our territory + ≥2 enemy colors) — **DONE, code-verified, but never visually confirmed by human eyes**
2. ≥5 real matches recorded + uploaded to HF — DONE then **DELETED as bad data**
3. Vision fine-tuned on real data: water ≥97%, me ≥90%, enemy ≥85%, UI ≥98% — **gate implemented & loud, but never passed** (runs v13 failed it honestly)
4. Policy ≥30% last-survivor win vs 12-player mixed/hard sim lobby
5. Real-game proof: bot wins 1 custom-scenario match as last survivor
6. README = live dashboard

---

## 4. Repo state (CLEAN — safe to build on)

`github.com/amerameryou1-blip/bot` — main branch, all fixes committed, working tree clean.
- **No data in the repo** (recordings/, weights/nn/ are gitignored and were deleted)
- **No secrets in the repo** (token-free; verified with grep)
- `.gitignore`: probe_*, bot_output, weights/nn, recordings, bot-train-nn.ipynb

```
run_bot.py                 live browser bot (Playwright) — nav, spawn detect, camera fix, ClickLoop
src/bot/camera.py          THE ZOOM FIX (wheel-event zoom toward self blob + numeric gate)
src/bot/ui.py              DOM helpers, strict Play matcher, lobby guard, enter_custom_match
src/bot/recorder.py        real-match recorder (256px frames + clicks + meta)
src/bot/planner.py         ClickPlanner (attack config now aggressive: density 42, ratio 1.5, pct 15)
src/bot/calibration.py     colors/blobs/OCR/leaderboard calibration
src/bot/click_loop.py      decision loop
src/bot/vision.py          segment() color-based scene parse
src/sim/game6.py           OFFLINE SIM on REAL maps, 8-15 bots, mixed skill, last-survivor win
src/sim/game5.py           older procedural sim
src/nn/model.py            TerritoryNet (85,533 params): seg(5) + localize + click-map(16x16) + kind + pct + value
src/nn/data.py             sim data collection
scripts/train_nn.py        THE TRAINER: collect → vision → clone → real → ppo → eval (stages)
scripts/label_real.py      real-frame auto-labeler (5-class per-pixel + click targets)
scripts/record_batch.py    sequential real-match recorder (crash-safe)
scripts/push_github.py     HF upload via private GitHub repo → Kaggle migration kernel → HF
scripts/launch_train.py    injects HF token into a TEMP copy, pushes GPU trainer as SCRIPT kernel, monitors
scripts/export_hf.py       model.safetensors → amer224/territorial-bot-nn
scripts/merge_worker_data.py  pull+merge sim shards from HF
scripts/collect_worker.py  Kaggle CPU worker: sim data shards → HF
scripts/rebuild_maps.py    real map extraction from screenshots (7 maps PASS: island, desert, pond, island_kingdom, middle_east, white_arena, black_arena)
kaggle-push/kaggle_train_nn.ipynb  GPU trainer notebook (token-free)
kaggle-push/migrate_kernel/        HF migration kernel (token-free; tokens injected at push)
weights/maps/*.npz         7 validated real maps (committed)
tests/                    47 passed / 0 failed (baseline green)
README.md                  live dashboard (updated through v13)
```

---

## 5. What is DONE & VERIFIED (evidence)

### ✅ THE ZOOM BLOCKER (the #1 blocker) — FIXED in code
- Found the game's zoom core in the 657KB inline JS: `du.zoom(ft,fg,fh){hs*=ft; hq=(hq+fg)*ft-fg; hr=(hr+fh)*ft-fh;}` — private, not on `window`.
- Wheel path: `canvasA.addEventListener('wheel', fb)` → `fb(eL)` → `eZ.fb(clientX,clientY,deltaY)` → `du.zoom()+fk()`; zoom centered on cursor; clamp `fj()`: `ft*ay ∈ [0.125, 1024]`; camera reset `iJ()` only on resize/menu.
- **Fix (`src/bot/camera.py`):** dispatch synthetic `WheelEvent`s on `canvasA` **at our blob's screen position** (zooms toward us), stepped 1.33× search until our blob = 1–6% of screen AND ≥2 distinct enemy clusters, numeric `verify_view()` PASS/FAIL, `ZOOM_LEVEL` env for forced zooms. Achieved ~80% PASS across 31 recorded matches.
- **CAVEAT for successor:** the zoom numbers were verified by code (pixel analysis), never by human eyes. **Look at `bot_output/cam_ok_*.png` style frames yourself** to confirm the view is actually good.

### ✅ Menu/lobby bugs fixed
- Menu items are `<div>/<p>` with emoji prefixes (`🗡️\nCustom Scenario`) — button-only scans missed them → bot sometimes joined the **multiplayer lobby** instead of a match. Fixed with all-element scan + strict emoji-stripped Play matcher + `is_in_lobby()` OCR guard + retry.
- **"Reset Scenario" exits the editor** (verified empirically) — flow does NOT click it; editor opens with correct defaults.

### ✅ Real-data pipeline (recordings → HF)
- 31 real matches recorded (~6,500 frames+clicks, 256px frames, clicks in original coords with `frame_scale`), uploaded to HF via **private GitHub repo → Kaggle migration kernel → HF** (HF blocks the sandbox's datacenter IP directly — proven via a Kaggle test kernel: 4/4 PASS from Kaggle's network).
- **ALL DELETED on 2026-08-08 as bad data** (user decision — see Section 6).

### ✅ Critical training bugs found & fixed
1. **Labeler mislabeled dark map pixels as UI** (dark navy ocean / dark land failed brightness floors) → NN could never learn water. Fixed: UI = known screen regions + bright text only, no darkness floor, neutral fallback. Water purity 98.5%, enemy class 2.5% → 11%.
2. **SILENT GATE BUG (the big one):** `classify_acc()` returns INT keys (0–4) but the gate checked STRING keys (`acc.get("water")` = always None) → **the vision gate ALWAYS "passed"** even when the model predicted 100% UI. v5 "completed" and shipped a collapsed model because of this. Fixed: int keys + the user's real gates (water≥97, me≥90, enemy≥85, ui≥98) + `raise RuntimeError` on failure. **v13 proved it works — the run FAILED LOUDLY instead of shipping garbage.**
3. **uint8→float input bug:** the real stage fed the model `d["rgb"]` (0–255 uint8) while sim pretraining used 0–1 → transfer wrecked. Fixed in v14 (`rgb_all = rgb / 255`).
4. **PPO loss=0.000 bug:** per-episode advantage normalization zeroes gradients when rewards are near-constant. Fixed: global batch normalization + reward×10 + entropy 0.05 + save best-eval (not last) checkpoint.

### ✅ GPU run history (all on Kaggle `amerameryou/bot-train-nn`, P100)
| Run | Result |
|---|---|
| v5 | Ran end-to-end, exported model to HF — but real stage collapsed to all-UI, silently "passed" (gate bug #2). NOT trusted. |
| v13 | Pulled 26 sessions/5,635 real frames (fixed labeler), vision hit **win-rate 0.33 / rank 2.83 (best ever)**, then real stage failed the gate → **stopped loudly at 3403s** (correct behavior). |
| v14 | Launched 2026-08-08 with all fixes (uint8 fix, capped sqrt-inverse weights [0.2,0.8], cosine LR 3e-5, sim-mixing, rare-frame oversampling, 12 epochs). **Status: STILL RUNNING as of handoff** — check/cancel on Kaggle. |

---

## 6. What was DELETED (clean slate)

- Local: `recordings/` (31 sessions), `weights/nn/*` (npz, models), probe artifacts — gone
- HF `amer224/territorial-bot-data`: `recordings/` + `realdata/` folders (7,338 files) deleted via kernel; **kept**: `screenshots/` (26 real map screenshots + index.json) and `shard_*.npz` (sim data)
- HF model repo `amer224/territorial-bot-nn` — deleted
- GitHub private transfer repo `amerameryou1-blip/bot-recordings` — deleted
- Kaggle datasets `territorial-bot-recordings-b1/b2/b3`, `territorial-bot-labels-b1` — **could not be deleted via API (Kaggle has no dataset-delete endpoint). They are private + harmless; delete in web UI if desired.**

---

## 7. THE NEW DIRECTION (decided with the user — build this)

**Train on thousands of SIMULATIONS against the sim's fighting bots** (not browser recordings). This is the standard "runs for hours and learns on its own" recipe:

```
        ┌─────────────────────────────────────────────────────┐
        │  CONTINUOUS RL LOOP (never stops)                   │
        │                                                     │
  ┌─────▼──────┐   trajectory shards   ┌──────────────────┐   │
  │ 5× Kaggle  │ ────────────────────▶ │ 1× Kaggle GPU    │   │
  │ CPU workers│   (states/actions/    │ PPO trainer loop  │   │
  │ play sim   │    rewards → HF)      │ (collect→train→   │   │
  │ matches    │ ◀──────────────────── │  save checkpoint) │   │
  └────────────┘   latest checkpoint   └────────┬─────────┘   │
                                                │             │
  watchdog re-launches kernels when sessions end (≈9-12h caps)│
  └───────────────────────────────────────────────────────────┘
```

- **Environment:** `src/sim/game6.py` — real maps (from your screenshots), 8–15 bots, mixed skill (easy/medium/hard), **these bots DO attack each other** (unlike the game's Very Easy), v8 rewards (win +5, kill×2, growth/2000, idle penalty), curriculum already in PPO (easy→hard, 2→10 enemies).
- **v1 (build now):** worker loop mode (play with latest checkpoint + ε-exploration → shards → HF), trainer loop mode (pull shards → PPO → save checkpoint + eval stats → HF), watchdog relauncher. Everything else exists.
- **v2 (self-play):** add policy-controlled opponents (older checkpoints from a pool) to `game6.py` — the "gets better forever" mechanism.
- **Eval gate:** ≥30% win vs 12-player mixed/hard before live-game validation.
- **Live-game validation (the vision part):** run `run_bot.py --record` with the final model, WATCH the frames, confirm the bot sees the map (zoom fix is in place) and actually wins as last survivor.

---

## 8. Credentials — ⚠️ ALL PLACEHOLDERS (real tokens live ONLY in the user's chat history)

| Service | Account | Token (placeholder — replace!) |
|---|---|---|
| GitHub | `amerameryou1-blip` | `ghp_***_REPLACE_ME` |
| Kaggle | `amerameryou` | `KGAT_***_REPLACE_ME` (set `~/.kaggle/access_token` + `kaggle.json`; `KAGGLE_CONFIG_DIR` pattern used) |
| Hugging Face | `amer224` | `hf_***_REPLACE_ME` (env `HF_TOKEN`) |

**Project is ending → the user should ROTATE these tokens** (their own rule #14). Never hardcode into public files — inject at push time (see `scripts/launch_train.py` + `scripts/push_github.py` patterns). GitHub secret-scan blocks token commits — that's how leaks got caught twice.

---

## 9. Data inventory (Hugging Face `amer224/territorial-bot-data`, private)

- `screenshots/<map>/*.png` — 26 real start-phase map screenshots + `index.json` ✅ KEPT (gold for map extraction)
- `shard_<map>_medium_s*.npz` — sim data from the 4 old CPU workers (island/mountains/desert/swamp/lakes, ~12,798 samples merged) ✅ KEPT
- `recordings/` + `realdata/` — DELETED (bad data)

---

## 10. Game mechanics knowledge (hard-won)

- Controls: double-click = claim land; Space = attack; W/S = attack% ±2%; D/A = ±0.5%; B = boat (3.125% tax); M = auto-attack; H = hide UI. Keybinds incl. "Zoom In"/"Zoom Out" exist in the key-action table.
- Economy: interest tick 0.56s, income tick 5.6s, soft cap 100 troops/px, hard cap 150/px, 7× boost first 107s, start 12px+512 balance, land attack tax 1.17%, defender 2:1.
- Editor: "Bot Difficulty → Uniform: Very Easy" — DOM row but clicks DON'T cycle it (canvas intercepts). "Map → Settings" opens the type panel (Procedural/Realistic/Custom — DOM `<p>` rows, radio moves on click) but the map-name list is canvas-rendered and flaky to click headlessly. Default = Island.
- Multiplayer lobby UNREACHABLE from datacenter IPs → Custom Scenario only.
- Zoom state lives in private vars (`hs`, `hq`, `hr`, `ay`, `b0`, `b2`); wheel listener on `canvasA` accepts synthetic events (no `isTrusted` check); `eZ.eT()` blocks wheel during zoom animations (space ticks ~0.7s).

## 11. Infrastructure lessons (all learned the hard way)

- **Sandbox resets every turn:** pip packages, playwright chromium, system libs (`libnspr4` etc.), `~/.kaggle`, git identity, `/tmp` all wiped. Reinstall each session. Workspace files under `/home/user` persist.
- **Workspace budget 128MB / 10k files** — push data out, delete locally, keep `.cache` out of the snapshot.
- **HF blocks this sandbox's IP** (HTTP 429 on everything) — route HF operations through Kaggle kernels (works from Kaggle's network).
- **Kaggle:** notebook kernels pushed via API do NOT auto-execute — use **script** kernels. Dataset sources attach only AFTER the dataset has a stable version (create → version → wait → attach; fresh datasets fail with "New Datasets cannot be attached in non-interactive sessions" — kagglehub path also flaky). No API delete for datasets/kernels.
- **GitHub:** public repo + secret scan → token commits blocked; use token-in-URL pushes with rotated tokens or credential helpers; private transfer repos work as a data ferry.
- **GPU budget:** ~36h/week, P100 (sm_60) needs torch 2.4.1+cu121 (auto-handled in the trainer); kernels capped ~9–12h → design loops to resume.

## 12. User preferences (honor every one)

1. Quality over quantity; real data first; NEVER 2M-sample procedural farms
2. Real-world fidelity: 8–15 enemies, vivid random colors, real maps
3. The click-head must know WHERE to click per-pixel (16×16 click-map is the deliverable)
4. WIN = LAST SURVIVOR (elimination), reward win +5 / kills / growth / idle penalty
5. FAIL LOUDLY — every stage prints concrete PASS/FAIL numbers, exits nonzero; a silent "COMPLETE" that didn't learn = hard fail
6. Think deep, write reasoning down; analyze → plan → verify → implement → re-verify
7. USE VISION constantly (this is why a vision-capable agent was requested)
8. Workspace hygiene — upload to HF, delete locally
9. Keep producing data; parallel processes; never let a crash end the stream
10. Fully autonomous — no babysitting; zoom automated
11. Difficulty = Normal when possible (couldn't be automated — honest note)
12. 2–4 matches per map, switch maps between batches
13. README = live dashboard with session table
14. Credentials rotate when project ends (now)
15. ~36 GPU-hrs/week on Kaggle, 5 concurrent CPU sessions; real-data fine-tunes + targeted PPO beat giant procedural runs

## 13. Recommended first steps for the successor

1. **Cancel/check the v14 GPU kernel** (may still be running — burning GPU hours): `kaggle kernels status amerameryou/bot-train-nn`; cancel in web UI if done.
2. Read `QUALITY_PLAN.md`, `HANDOFF_BEST.md`, `ATTACK_META.md` in the repo.
3. Reinstall env (torch CPU, playwright chromium, tesseract, kaggle CLI, git identity) — sandbox resets wiped them.
4. Run `PYTHONPATH=src python3 -m pytest tests -q` → expect 47 passed.
5. **Build the continuous sim RL loop (Section 7, v1):** worker loop-mode + trainer loop-mode + watchdog. The PPO/vision/gates are fixed and committed.
6. Get eval win-rate vs 12-player mixed/hard lobby climbing (target ≥30%).
7. Then, with YOUR vision: validate in live matches — run `python3 run_bot.py --record --games 3 --minutes 3`, LOOK at frames, confirm zoom/view/enemy visibility, and get the last-survivor win.

---

*End of handoff. The repo is clean, honest, and ready. Good luck — the bot is one good vision module and a continuous RL loop away from winning.*
