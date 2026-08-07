# QUALITY-FIRST PLAN — beat territorial.io as LAST SURVIVOR, with real data

> Written 2026-08-07 after full verification. This is the PLAN. No code changes until you say go.
> Principle you set: **highest quality over quantity. Most real data first.**
> Evidence this plan is based on: the v7 run's own log (COMPLETE, verified below), the 26 real
> screenshots on HF, and the live bot's first real win (rank #1 but expansion-only, no attacks).

---

## 0. Verified status (what we know is TRUE right now)

| Item | State |
|---|---|
| Tests | 43 passed, 1 skipped (`pytest tests`) — all green |
| v7 training (Kaggle GPU `bot-train-nn`) | **COMPLETE**, no silent failures |
| P100 CUDA fix | works — torch 2.4.1+cu121 auto-install, "CUDA OK", full GPU run |
| Vision training | learned (loss 2.06 → 1.24) |
| Clone training | learned (loss 9.08 → 5.57) |
| PPO survival | learned vs easy (alive 0.00 → 0.25–1.00, curriculum escalates) |
| **PPO winning** | **FAILED — eval vs hard bots: alive 0.00, win-rate 0.00, rank 2.83/4 every time** |
| Training data size | only 3,823 samples (too small, and 100% sim-generated) |
| Real screenshots on HF | 26 start-phase shots, 23 maps, organized into `screenshots/<map>/` + `index.json` |
| HF dataset | private, organized; ~9,443 sim samples across 5 maps |
| Live bot | won 1 real match (rank #1) but only expanded — never attacked |
| Workspace | clean (2.5 MB, 199 files); v7 model persisted in `weights/nn/model.pt` |

**The one honest verdict:** the whole pipeline works mechanically, but the bot has never
*attacked well enough to be the last survivor*. That is the problem this plan exists to fix —
with real data, real maps, real enemies, real colors, and a pixel-level click brain.

---

## 1. What "highest quality" means here (the bar)

Quality = **fidelity to the real game**, ranked by how much each input matches reality:

1. **Real game frames** (screenshots, recorded matches) — the only data that is literally the real game.
2. **Maps reconstructed from real screenshots** — real coastlines, real water, real dimensions, real land%.
3. **Enemies that look and act like real players** — many of them, distinct colors, varied aggression.
4. **Clicks a strong player would actually make** — real recorded clicks as ground truth.
5. Sim-generated procedural fill — used ONLY to add volume *after* the real stuff exists.

**Rejected because it's quantity without quality:** generating millions more procedural
110×140 maps with 3 bots. The v7 run already proved more-of-the-same doesn't make it win.
The bottleneck is *realness*, not *count*.

---

## 2. The data pyramid (what we collect, in priority order)

```
                    ▲  QUALITY
                   / \
                 /     \     1. REAL MATCH RECORDINGS  (rarest, most valuable)
               /         \      frames + every click + leaderboard + map
             /  mid-match \  2. MID-MATCH SCREENSHOTS   (~80–150, user takes them)
           /   screenshots   \   multi-enemy, different colors, leaderboard visible
         /  start-phase shots  \  3. START-PHASE SCREENSHOTS  (have 26, 23 maps)
       /  real-coastline sim maps \ 4. REBUILT REAL MAPS in sim (from #3)
     /  realistic sim play  \       5. SIM PLAY: 8–15 enemies, real colors, boats
   /   procedural fill       \      6. PROCEDURAL (only to pad volume)
  ─────────────────────────── ▼ QUANTITY (cheap, infinite)
```

### 2.1 Real match recordings (the crown jewel — enables "where to click per pixel")
The live bot already plays real custom-scenario matches. Extend `run_bot.py` to **record**:
- every frame (compressed, ~5 fps is plenty),
- **every click** the bot makes (position + time + whether it claimed land / attacked / banked),
- the leaderboard strip (for self-detection),
- the map + dimension (known at match start).

Output: `(frame, click_target, click_kind)` triplets. This is *exactly* the supervised data
needed for the click-map head: **"given this frame, where would a strong player click?"**
Real clicks, real maps, real colors — nothing generated. ~20–40 matches → ~100–200k frames.

### 2.2 Mid-match screenshots (user action — highest leverage per minute of your time)
While playing real matches (custom scenario, **Colors: Random**, Player Count 10+):
- screenshot every ~30–60 seconds, full-screen, leaderboard visible,
- a few different maps per session, covering early/mid/late game,
- aim for **~80–150 total** over the next days. Each gives me: me vs enemy territory colors,
  water, neutral land, UI, leaderboard → auto-labels for vision fine-tuning.

### 2.3 We already have 26 start-phase screenshots (23 maps) — done.
Used next for map reconstruction (below).

### 2.4 Rebuilt real maps in the sim — from the 26 screenshots
Every start screenshot has the FULL map visible + OCR'd dimension + land/water/mountain %.
Plan: extract the terrain mask (water = blue hue band, land = beige/green, mountain =
gray/brown), downsample to sim resolution, **validate land% against the OCR'd number (±1%)**
and coastline IoU vs the screenshot mask (≥90%). Result: the sim's maps ARE the real maps —
same water, same continents, same proportions (Desert 82.6% land, Island Kingdom 87.4% water,
World 70.4% water, Cliffs with 16% mountains...). This is the "generate water" fix done right:
real water from real screenshots, not procedural blobs.

### 2.5 Realistic sim play (the RL arena)
- **8–15 enemies** (real game is 10–16 players), not 3.
- **Real distinct colors** from the game's palette (the vivid colors in the screenshots).
- **Varied behaviors**: greedy full-send expanders, balanced attackers, defensive campers —
  a real lobby is a mix, and the bot must handle all of them.
- **Water mechanics**: boats exist in the real game (B key, 3.125% tax). Sim gets a boat
  toggle for water-crossing so Island Kingdom / World are playable like reality.
- Enemies attack EACH OTHER too (FFA), exactly like a real lobby.

---

## 3. The "which pixels are me" fix (vision that works in the real game)

Goal: from a real frame, classify every pixel: `me / enemy / neutral / water / UI`.

1. **Self-color anchor:** in custom scenario with Colors: Random, our territory color is
   set by us in the editor (proven to work). Also, the leaderboard shows each player's
   color next to their name — the bot OCRs its own (highlighted) row → ground-truth color.
2. **Auto-label real frames** (recordings + mid-match screenshots):
   - me = pixels ≈ our anchored color (morphological cleanup),
   - enemy = other vivid saturated blobs (unique-color clustering),
   - water = blue hue band (calibrated per map from the start screenshot),
   - UI = fixed regions (leaderboard, buttons, bars) detected by position + edge stats,
   - neutral = unclaimed land-colored pixels.
3. **Fine-tune the CNN on real frames** (mix 60% real + 40% sim to keep generalizing),
   monitor per-class pixel accuracy on held-out real screenshots:
   **water ≥97%, me ≥90%, enemy ≥85%, UI ≥98%** — loud gate.

---

## 4. "Where to click, per pixel" (the action head redesign)

Current: 64×64 input → 16×16 click grid + kind(expand/attack/bank) + pct + value.
That's too coarse for "identify where to click each pixel". New design:

```
frame (64×64 or 128×128)
   │
   ├─ vision head ────────────► per-pixel class map (me/enemy/neutral/water/UI)
   │
   └─ policy head (PPO + clone):
         coarse heatmap 16×16  ─┐
         attention refine ──────┼─► precise click point (x,y) at FULL resolution
         kind + amount ─────────┘   + kind (expand/attack/bank) + troop %
```

- **Supervised ("clone") loss** on the click head with REAL click labels from recordings
  (heatmap = Gaussian around the real click) + sim planner labels where real data is thin.
- **RL ("PPO") loss** continues on top for the long-horizon "attack now vs expand now"
  decision.
- **Water-aware:** the vision water mask feeds the click head — it must never pick water
  (unless boat mode is on). Learned, not hard-coded.

This directly gives the bot pixel-level click targets in the real game.

---

## 5. Fixing "never attacks" — reward & curriculum v8 (the v7 lesson)

v7 taught us: survival bonus + easy-start ⇒ bot survives easy bots but still loses every
eval vs hard (rank 2.83/4, 0 wins). Diagnosis:

| v7 problem | v8 fix |
|---|---|
| +0.02/tick survival dominates; hiding farms it | shrink to +0.005; add **territory-growth reward** (net area delta), **kill bonus** (+2), **attack-success bonus**, small **idle penalty** |
| Win +1.5 too weak vs survival stream | win = **+5** (last-survivor is the goal) |
| Curriculum escalates by alive-rate only | escalate by **last-survivor win-rate** AND **enemy count**: 2 easy → 4 → 8 → 12, skill easy→medium→hard |
| Only 4 players | **10–16 player FFA** from the start of the phase |
| Eval = hard bots only, 4-player | eval = 12-player mix; **ship gate: ≥30% last-survivor win-rate** |

---

## 6. Water & boats (phased in AFTER land-wins works)

1. First get expand/attack/bank land gameplay winning (phases A–E below).
2. Then add boats: sim boat toggle (3.125% tax like reality), policy boat action,
   planner labels for when crossing water wins (e.g., Island Kingdom, World).
3. Vision already has the water class; the click head learns to target water tiles
   only in boat mode.

Don't dilute the core learning with boats before the core works.

---

## 7. Training schedule (your ~36 h/week GPU)

| Phase | What | GPU time |
|---|---|---|
| **A. Real data week** | rebuild 23 real maps from screenshots; you shoot ~80–150 mid-match screenshots; bot records ~10 real matches; build the auto-labeler | 0 (CPU/sim) |
| **B. Vision fine-tune** | fine-tune CNN on real frames (60/40 mix); check per-class gates | ~2–3 h |
| **C. Sim arena v8** | 10–16 enemies, real colors/maps/water, new rewards | 0 (CPU/sim) |
| **D. PPO v8** | curriculum 2→12 enemies, easy→hard; eval every N rounds | ~20–30 h |
| **E. Clone on real clicks** | behavior-clone the click head on recorded real matches, then PPO a bit more | ~3–5 h |
| **F. Real test** | bot plays real custom scenario; record; iterate A→F | runtime only |

Priority order (quality-first): **B before D** (a blind bot can't learn to attack well);
**real clicks (E) before scale-out** (real click data beats more sim episodes).

---

## 8. Runtime weight budget (stay lightweight)

- Keep the model ≤ ~150k params (85k now + refine head). 64×64 input at 5–10 Hz on CPU is
  already proven; the refine head is tiny. Per-pixel click map is computed hierarchically
  (16×16 coarse → refine), not a giant dense output, to stay fast.
- Ship as safetensors to `amer224/territorial-bot-nn` (export script ready).

---

## 9. Loud quality gates (fail loudly, never silent)

1. **Map fidelity:** land% within ±1% of OCR'd real stat; coastline IoU ≥90% vs screenshot.
2. **Vision (real frames):** water ≥97%, me ≥90%, enemy ≥85%, UI ≥98% pixel accuracy on
   held-out real screenshots.
3. **Policy:** ≥30% last-survivor win vs 12 hard bots in sim before a real-game test.
4. **Real game:** win at least 1 real custom-scenario match as last survivor (not rank 1 by
   area — by ELIMINATION).
5. Every stage prints PASS/FAIL numbers. FAIL stops the run. (Existing rule, keep it.)

---

## 10. What I need from YOU (this week)

1. **~80–150 mid-match screenshots** — real matches, Colors: Random, 10+ players,
   full-screen with leaderboard, every ~30–60s, across a few maps. (The single biggest
   quality lever — this is the "most real ones" you asked for.)
2. Let the bot record **~10 real matches** (I'll enable recording in `run_bot.py` when you
   say go — it plays the same custom scenario it already wins).
3. Decide: make `territorial-bot-data` public (sim + screenshots are fine to share;
   recordings contain your nickname — keep private or strip names).
4. Confirm the ship gate (≥30% vs 12 hard bots) before real-game testing.

## 11. Explicitly NOT doing (until gates pass)

- ❌ 2M-sample worker farms of procedural sim data — quantity without quality.
- ❌ Boats before land-wins works.
- ❌ Making the model bigger/fancier before the data is real.
- ❌ Shipping a model that loses every eval — no "silent COMPLETE".

---

*Next step when you say GO: Phase A — map reconstruction from the 26 screenshots +
recording hooks + auto-labeler. Everything in this doc is code-ready to start.*
