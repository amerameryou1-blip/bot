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
**Status:** RUNNING — last checked 2026-08-07 ~14:20 UTC

How to watch it yourself:
1. https://www.kaggle.com/code/amerameryou/bot-train-nn
2. The final cell prints the win-rate vs hard bots, and uploads weights to
   Hugging Face `amer224/territorial-bot-nn` (if HF_TOKEN secret is set)
3. The live bot (`run_bot.py`) auto-loads the HF model and plays with the NN brain

Curriculum logic: PPO starts vs MEDIUM bots; when eval win-rate > 65% it
upgrades to HARD; if < 30% it drops back. This is the easy→medium→hard plan.
