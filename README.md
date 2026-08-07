i forgot to tell you before i left there ri smultible maps Options
🟢 Procedural Map

⚪ Realistic Map

⚪ Custom Map

Map
⚪ White Arena

⚪ Black Arena

🟢 Island

⚪ Mountains 1

⚪ Desert

⚪ Swamp

⚪ White Plains

⚪ Cliffs

⚪ Pond

⚪ Halo

⚪ Island Kingdom

⚪ Mountains 2 you can chnge them by clikcing on map setting

---

## 🔄 LIVE TRAINING STATUS (auto-updated by the bot)

**Training run:** Kaggle GPU notebook `amerameryou/bot-train-nn` (T4 GPU, Internet on)
**Pipeline:** collect (CPU-parallel, medium bots) → vision (GPU) → clone (GPU) → PPO (GPU, curriculum medium→hard) → HF upload
**Status (v7):** main trainer **COMPLETE** ✅ (verified 2026-08-07) — 4 data workers COMPLETE ✅

**v7 run result (from the kernel log — the honest numbers):**
- Pipeline ran end-to-end on the P100: torch 2.4.1+cu121 installed itself, CUDA OK,
  collect 3,673 samples → merged 1 HF shard → vision (loss 2.06→1.24) → clone
  (loss 9.08→5.57) → 100 PPO rounds → eval. No silent failures.
- ✅ **It learned to SURVIVE vs easy bots**: PPO alive-rate went 0.00 → 0.25–1.00,
  curriculum auto-moved easy→medium→hard (multiple times).
- ❌ **It still CANNOT WIN**: every eval vs hard bots = alive 0.00, win-rate 0.00,
  rank 2.83/4 (finishes 3rd, never 1st). Training data was tiny (3,823 samples)
  and the survival bonus teaches hiding, not attacking.
- Lesson: survival reward fixes "dying instantly" but not "never attacks".
  → next version (v8, see QUALITY_PLAN.md) reshapes rewards around kills/territory
  growth/win + many enemies (8–15) + real-map sim + real-screenshot vision.

**Reusable data saved to Hugging Face (for future bigger models):**
`amer224/territorial-bot-data` (dataset repo, private) — pull anytime with
`scripts/merge_worker_data.py` (HF_TOKEN):
- island: 2,235 samples · mountains: 2,445 · desert: 2,356 · swamp: 2,407
- ~9,400 samples / ~120MB across 4 map types + the lakes data from the main run
- Regenerate/extend anytime by running the worker notebooks again (different seeds/maps)


**v7 fixes (from the v6 run that finished but didn't learn — alive was 0.00 the whole time):**
- **Reward fix:** added a per-step survival bonus (+0.02) + bigger growth signal.
  Before, the only signal was -1 on death → no positive gradient → never learned.
- **Curriculum fix:** PPO now starts vs EASY bots and escalates by ALIVE RATE
  (easy→medium at >35% survival, medium→hard at >60%). Before it started at
  medium and the random policy died instantly every episode.
- **Proved locally:** alive went 0.00 → **1.00** in round 1 (easy) with auto-upgrade.

The v6 run DID prove the whole GPU pipeline works end-to-end (collect→vision→clone→
100 PPO rounds→eval on GPU) — it just needed the reward/curriculum fix to actually learn.

| Session | Role |
|---|---|
| `bot-train-nn` (GPU) | main trainer: collect → vision → clone → PPO → eval |
| `worker-island` (CPU) | data slice: island map → HF `amer224/territorial-bot-data` |
| `worker-mountains` (CPU) | data slice: mountains map → HF |
| `worker-desert` (CPU) | data slice: desert map → HF |
| `worker-swamp` (CPU) | data slice: swamp map → HF |

The main session pulls + merges all worker shards (via `scripts/merge_worker_data.py`)
before vision/clone, so the CNN trains on **5 different map layouts** — fixing the
single-map overfit I flagged when you showed me the map selector.

**The vision fix (how the bot learns which pixels are IT in the real game):**
1. Bot sets its OWN vivid color in the editor (RGB fields — we verified this works)
2. Records real matches (custom scenario)
3. Auto-labels real frames using: leaderboard swatch + spawn aura + movement
   consistency (my territory persists/grows; enemies move)
4. Fine-tunes the vision CNN on real frames → learns real maps/colors/UI/borders
   → no more "can't tell which one is me" even with camouflage colors

**Fixes shipped in v6 (from the failed runs):**
- P100 sm_60 support: notebook auto-installs torch 2.4.1+cu121 (last build with
  P100 kernels); falls back to CPU if still broken (model is only 85k params)
- Device-safe trainer: real CUDA sanity test at startup (torch.cuda.is_available
  lies for P100), device-aware tensors (fixed the cuda-vs-cpu crash)
- PPO shape bug fixed; bounded-memory rollouts (uint8 frames, act every 4 ticks)
- Stage failures now STOP LOUDLY (never a silent COMPLETE)

How to watch it yourself:
1. https://www.kaggle.com/code/amerameryou/bot-train-nn
2. The final cell prints the win-rate vs hard bots, and uploads weights to
   Hugging Face `amer224/territorial-bot-nn` (if HF_TOKEN secret is set)
3. The live bot (`run_bot.py`) auto-loads the HF model and plays with the NN brain

Curriculum logic (v7): PPO starts vs EASY bots and escalates by how well it
survives — easy→medium when alive-rate > 0.35, medium→hard when > 0.6,
drop back a level if < 0.3. Plus a per-step survival bonus (+0.02) and win
bonus (+1.5) so the network learns to LIVE first, expand second. (v6 ran the
whole 100 rounds with alive 0.00 — the medium-only, no-survival-bonus reward
never taught it anything. v7's easy-start + survival reward is the fix.)

## Real-game screenshots (map data)

26 real screenshots uploaded by the user, now organized on HF in
`amer224/territorial-bot-data` under `screenshots/<map>/` (was: 26 loose
files at repo root). All are the "Choose your start position!" phase = full
map visible + on-screen stats. `screenshots/index.json` maps every file to
map name, real dimension, and OCR'd land/water/mountain %.

Coverage (23 distinct maps): island, white_arena, black_arena, mountains(2),
desert, swamp, white_plains(2), cliffs, pond, halo, island_kingdom, europe,
world(2), caucasia, africa, middle_east, scandinavia, north_america,
south_america, asia, australia, british_isles, mare_nostrum.

Use: rebuild sim maps at REAL dimensions/coastlines (per-map land% from OCR
is the validation target), and as the seed for real-frame vision fine-tuning.


## 🧪 v8 build (quality-first — built 2026-08-07 while you slept)

**Real maps from real screenshots** (`scripts/rebuild_maps.py`): extracted
coastlines/water/mountains from the 26 start-phase screenshots. **7 maps
validated PASS** (land/water/mountain% within ±5 of the in-game OCR stats):
island, desert, pond, island_kingdom, middle_east, white_arena, black_arena.
The other 19 fail LOUDLY (ambiguous gray-land in those screenshots) — never
silently shipped. Preview + ascii stored in `weights/maps/`.

**Recorder** (`src/bot/recorder.py`, `run_bot.py --record --games N --upload`):
records every frame + every click the bot makes + self/enemy colors + map
name (OCR) per match, uploads to HF `amer224/territorial-bot-data/recordings/`.
This is the real-click ground truth ("where to click per pixel").

**Auto-labeler** (`scripts/label_real.py`): turns recordings + screenshots into
5-class per-pixel labels (water/neutral/me/enemy/ui) + click targets. Verified
locally.

**Sim v8** (`src/sim/game6.py`): real maps, 8-15 enemies, mixed-skill lobbies,
vivid real colors, kill tracking. **Trainer v8** (`scripts/train_nn.py`):
reward = growth + kill ×2 + win +5 (last survivor!) − idle penalty;
curriculum escalates BOTH skill (easy→medium→hard) AND enemy count (2→…→10).
`stage_real` fine-tunes vision on real frames + clones clicks on real clicks.
Verified: PPO smoke run works on CPU, curriculum fires.

**Kaggle notebook updated** (`kaggle-push/kaggle_train_nn.ipynb`): pulls real
recordings from HF → auto-labels → vision+clone+real → PPO v8 on real maps.
