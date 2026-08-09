# 🔍 FILE-BY-FILE REVIEW GATE (before any GPU minute is spent)

User rule (2026-08-09): "make you review each file independently before we move
to train on GPU". This doc IS that review. Every file that feeds the GPU
teacher run is audited: purpose, verified-by, status, open issues.

Legend: ✅ smoke-tested locally · 👁 verified with vision · ⚠ open issue

---

## 1. `src/nn/model_v2.py` — the brains ✅
- TeacherV2 = 100.3M total, sparse MoE (8 experts, top-2 → ~40-45M active).
  StudentV2 = 9.3M dense. I/O: (B,6,128,128)=[frame,diff] + (B,8) numbers.
- Verified: forward shapes for all heads; param budgets printed & asserted.
- ⚠ MoE dispatch is a python loop (correct, slow). Fine for supervised GPU
  run (batched per expert); MUST vectorize before PPO rollouts.
- ⚠ Width knob = one line each (`ex=576`, `px=`): 100M↔60M fallback preserved.

## 2. `src/sim/game6.py` — the gym ✅
- `render_fast` vectorized full-res rgb+labels (no python loop) — output
  compared by eye against live-game frames (same palette).
- `numeric_ctx` = the 8 leaderboard numbers (balance, fracs, red flag,
  tick, kills) — exact in sim, OCR-sourced in live bot. Consistent scale
  (log/frac) so sim→real transfer holds.
- ⚠ labels: 4 classes (water/neutral/me/enemy); teacher head has 5 (ui
  unused in sim, used by real labels). CE is compatible; documented.

## 3. `scripts/rl_loop.py` (v2 worker) ✅
- Records 128px bundles + labels + nums + actions; shard_v2 format
  (uint8 rgb, uint8 lab, f32 nums/reward, lens). Local smoke: 27 shards ok.
- Actions come from current best net at 64px (farmer role) — intended.
- ⚠ `logp` stored as constant -2 (placeholder). Supervised stages ignore
  logp; teacher PPO stage MUST record true logp (TODO before ppo stage).
- Storage: v2 shards ~3× v1 size; V2 worker cap must stay ≤ ~400 shards.

## 4. `scripts/train_v2.py` — teacher/student trainer ✅
- `sup`: seg CE (free sim labels) + click-clone + kind + pct. Smoke: loss
  finite, teacher.pt saved on CPU.
- `distill`: teacher→student KL (seg/click/kind) + hard labels. Smoke:
  student.pt saved.
- NaN watchdog (HybridOpt) on both. GPU path = same code (device-agnostic).
- ⚠ bs=16@128px needs ≥16GB RAM for full epochs → Kaggle VM (30GB) or P100.

## 5. Data audit — `scripts/audit_data.py` ✅ / ⚠
- v1 recordings: 4 bad purged, 4 kept w/ UI-click filter. v1 shards clean.
- ⚠ TODO: extend audit to v2 shards (rgb range, lens sum, kind/cell range)
  and run before GPU sup. (Trivial — same checks, new folder.)

## 6. Ops — launch/watchdog ✅
- v2 workers get own slugs (`rl-v2-worker-1/2`), watchdog relaunches them
  with V2 boot. CPU cap respected (fleet ≤5 concurrent).
- GPU kernels ONLY via explicit launch scripts; no auto-GPU anywhere.

---

## 🚦 GPU RELEASE CHECKLIST (teacher supervised pretrain)
- [x] model shapes/params smoke
- [x] v2 worker farms locally
- [x] sup + distill smoke on CPU
- [ ] v2 shards on HF ≥ ~1.5GB (workers filling now)
- [ ] v2 shard audit PASS
- [ ] this doc re-read & still green
→ then: ONE GPU run, supervised only (PPO later, separate review).
