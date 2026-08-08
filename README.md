# 🤖 Territorial.io Autonomous Bot — AureliaBot

Autonomous agent that plays **territorial.io** Custom Scenario (offline vs bots) and
must WIN AS **LAST SURVIVOR** (by elimination — NOT biggest area).

> **Readme = live status dashboard.** Session table below is updated after every run.

---

> ## ⚡ OVERNIGHT SESSION PROGRESS (2026-08-07→08, autonomous)
> - ✅ **Zoom blocker FIXED** — numeric proof: self blob 0.3-0.5% → 1.2-1.9% in window, ≥2 enemies; ~80% PASS by batch 3-4
> - ✅ **25 real matches recorded** (~5,500 frames + clicks) → HF `recordings/` (b1-b4 verified; b5 recording)
> - ✅ **v5 GPU run completed end-to-end** (P100, torch 2.4.1+cu121 auto-fixed): collect→vision→clone→real→PPO→eval→exported model.safetensors to `amer224/territorial-bot-nn`
> - ❌ **v5 quality FAILED (caught loudly, NOT shipped):** the real-stage fine-tune collapsed segmentation to all-UI. Root causes found & FIXED:
>   1. **Labeler bug**: dark map pixels (dark navy ocean, dark land) were labeled UI → the NN literally could not learn water/land. Rewrote `classify_frame`: UI = screen regions + bright text only, no darkness floor, neutral fallback. Water purity now 98.5%, enemy class 2.5%→11%.
>   2. **Silent gate bug**: `classify_acc` returns INT keys but the gate checked STRING keys → `acc.get("water")` was always None → **gate always PASSED**. Fixed to int keys with the user's real gates (water≥97%, me≥90%, enemy≥85%, ui≥98%) and it now RAISES on failure.
>   3. Class weights: fixed [1,1,2.5,2.5,1] → sqrt-inverse-frequency; LR 3e-4→5e-5.
> - 🚀 **v6 GPU training RUNNING** with all fixes + 26 HF sessions (b1-b4)
> - 📦 Data flow: sandbox records → private GitHub repo → migration kernel → HF (HF IP-blocked from sandbox; Kaggle dataset attach flaky)

---|---------|-----|------|--------|--------|----------|---------|----------------|
| 1 | 20260807-202555-355808 | Island | 1 | 120 | 120 | FAIL | 4 | no |
| 2 | 20260807-202738-3742a0 | Island | 1 | 220 | 220 | FAIL | 5 | no |
| 3 | 20260807-203007-498b7f | Island | 0 | 260 | 260 | PASS | 3 | no |
| 4 | 20260807-203239-add512 | Island | 1 | 213 | 213 | FAIL | 0 | no |
| 5 | 20260807-203516-a4fcc7 | Island | 0 | 218 | 218 | FAIL | 6 | YES |
| 6 | 20260807-203752-fe3dfa | Island | 1 | 260 | 260 | FAIL | 3 | no |
| 7 | 20260807-204028-195e2a | Island | 0 | 218 | 218 | FAIL | 6 | no |
| 8 | 20260807-205446-cafa37 | Island | 1 | 220 | 220 | FAIL | 3 | no |
| 9 | 20260807-205719-9a40dc | Island | 0 | 260 | 260 | PASS | 2 | no |
| 10 | 20260807-205949-e36d00 | Island | 2 | 213 | 213 | FAIL | 1 | no |
| 11 | 20260807-210228-5b470d | Island | 1 | 220 | 220 | FAIL | 3 | no |
| 12 | 20260807-210507-4f4b9d | Island | 0 | 219 | 219 | FAIL | 5 | YES |
| 13 | 20260807-210749-b91c32 | Island | 2 | 220 | 220 | PASS | 3 | no |
| 14 | 20260807-212055-13fc79 | Island | 1 | 220 | 220 | PASS | 3 | no |
| 15 | 20260807-212343-e33537 | Island | 0 | 210 | 210 | PASS | 2 | no |
| 16 | 20260807-212618-ed0809 | Island | 2 | 0 | 0 | FAIL | 0 | no |

### 🔬 Honest notes / open items
- **Map switching:** the editor's map type selector (Settings → Realistic Map)
  is clickable via DOM, but the map-name list is canvas-rendered and flaky to
  click headlessly → recordings are on the default map (**Island**) for now.
  Revisit with OCR-driven canvas clicking.
- **Difficulty:** editor shows "Uniform: Very Easy" and clicks do not cycle it
  (canvas intercepts) → matches run at Very Easy; no combat data from enemy-vs-
  enemy fights yet (they don't attack each other on easy).
- **Enemy detection** uses color heuristics (brightness floor, UI/self/dominant-
  background filters) — terrain-colored enemies can be missed; the NN
  segmentation is the real fix (fine-tune on these recordings).
- **HF upload** is blocked from this sandbox's IP (HTTP 429 on everything) →
  recordings migrate to HF via a private Kaggle kernel (the same route the
  existing workers use). See `scripts/` + session log.

---

## MISSION & ARCHITECTURE

```
sandbox (this repo)                    Kaggle                         Hugging Face
├─ run_bot.py (live Playwright bot)    ├─ bot-train-nn (GPU trainer)  amer224/territorial-bot-data
├─ scripts/record_batch.py (recorder)  ├─ worker-{island,desert,...}  ├─ recordings/<session>/
├─ scripts/label_real.py (auto-label)  │   (CPU data farmers, 5×)     ├─ screenshots/
├─ scripts/train_nn.py (stages)        └─ migrate-* kernels (HF link) └─ shard_*.npz (sim)
└─ weights/nn/ (trained model)
```

- Win condition: **last survivor** (elimination), reward +5 win, kill bonus,
  growth, idle penalty (v8, see QUALITY_PLAN.md).
- Real-data-first: recordings → auto-label → fine-tune vision+click policy →
  PPO on Kaggle GPU → export to HF (`scripts/export_hf.py`).
- Credentials: env vars / private kernels only. Never in public files.

## REPO MAP
- `run_bot.py` — live bot entry (nav → spawn → camera fix → ClickLoop → report)
- `src/bot/camera.py` — the zoom fix (this session's main deliverable)
- `src/bot/ui.py` — DOM menu helpers + lobby guard
- `src/bot/recorder.py` — real-match recorder (downscaled frames + clicks)
- `src/sim/game6.py` — offline sim on REAL extracted maps, 8–15 bots
- `scripts/` — label_real.py, record_batch.py, train_nn.py, rebuild_maps.py, export_hf.py
- `weights/maps/*.npz` — 7 validated real maps
