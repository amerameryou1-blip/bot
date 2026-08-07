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
**Status (v7):** main trainer RUNNING — 4 data workers COMPLETE ✅ — last checked 2026-08-07 ~16:30 UTC

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

Curriculum logic: PPO starts vs MEDIUM bots; when eval win-rate > 65% it
upgrades to HARD; if < 30% it drops back. This is the easy→medium→hard plan.
