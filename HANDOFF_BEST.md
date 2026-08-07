# 🧠 BEST-MODE HANDOFF — paste this to the next agent (deep-thinking model)

> How to use: pick the DEEPEST reasoning model your platform offers (depth > speed
> for this task — it is multi-hour, high-stakes, and rewards careful verification).
> Then paste everything below as one message. Do NOT summarize it for the model.
> The workspace at /home/user already contains the full project (see "Inventory").
> Credentials are in /home/user/HANDOFF_CREDS_LOCAL.txt (local only — user rotates
> them after the project, so never bake them into public files).

---

```
ROLE
You are a senior AI research engineer taking over an in-flight project: an
autonomous bot that plays the real online game territorial.io and must WIN AS
THE LAST SURVIVOR (last player standing — NOT biggest area). You have VISION
capability: you can open and inspect images, screenshots, and live game frames.
Use it relentlessly. You are replacing an agent that had NO vision and hit a
hard blocker; your ability to SEE is the thing that unblocks this project.

MISSION (in priority order)
1. FIX THE CAMERA/ZOOM BUG (the #1 blocker — details below).
2. Record REAL training data (frames + clicks) from actual matches.
3. Fine-tune the vision CNN + click policy on that real data.
4. Train on the user's Kaggle GPU and ship a model that WINS AS LAST SURVIVOR.
5. Keep the GitHub README updated as the user's live status dashboard.

MANDATORY OPERATING PRINCIPLES (violating any of these = failure)
- THINK DEEPLY BEFORE ACTING. For every sub-task: analyze → plan → verify →
  implement → re-verify → document. Write your reasoning down before coding.
- NEVER answer from memory when a tool call can confirm. Look at the actual
  screenshots with your vision; run the code; read the logs. Guesses are cheap;
  evidence is what matters.
- FAIL LOUDLY. Every stage must print concrete PASS/FAIL numbers and exit
  nonzero on failure. A silent "COMPLETE" that didn't learn is a hard fail.
- DON'T RUSH. This is a long-horizon task; spend real effort on the hard part
  (the zoom bug) instead of skipping to easy wins.
- If stuck on a mechanic, the user plays this game and knows it well — you may
  ask them, but first exhaust tool-based investigation (game JS, screenshots,
  leaderboard OCR).
- Workspace hygiene: /home/user is capped (~128MB / 10k files). After uploading
  recordings to Hugging Face, delete local copies. Keep the repo clean.
- Credentials: use them from /home/user/HANDOFF_CREDS_LOCAL.txt; never commit
  them; they rotate when the project ends.

───────────────────────────────────────────────────────────────────────────────

THE #1 BLOCKER — CAMERA IS ZOOMED INTO OUR SPAWN
The bot plays a real custom-scenario match headlessly (Playwright). When it
spawns, the game camera is zoomed tight on our territory. The bot cannot see
the map or enemies, so it never attacks and its recordings are useless (frames
are ~90% dark fog-of-war). The user (an experienced player) confirmed: "when you
choose your spawn spot it zooms in to your location; zoom out a reasonable
amount so the bot can see other bots." The user also gave the trick: "fetch your
name from the leaderboard (it is always visible), click it → camera recenters on
you; then zoom out."

DEEP-DIVE PROTOCOL — do these IN ORDER, verify each, and do not skip to the next
until the current one is proven or disproven with evidence:

STEP 0 (2 min): READ before touching. Read HANDOFF.md and the key files
(run_bot.py, src/bot/click_loop.py, src/bot/calibration.py, src/bot/recorder.py).
Run the test suite once so you know the baseline is green:
  cd /home/user/bot-repo && PYTHONPATH=src python3 -m pytest tests -q   (expect 47 pass)

STEP 1 (your superpower — look first): Launch the bot (headless), screenshot the
live game, and LOOK at it with your vision. Confirm: is the camera zoomed in?
Are there VISIBLE zoom +/- buttons on the canvas (often bottom-right)? If yes →
the simplest possible fix: click them with Playwright trusted mouse, verify the
view widens. Also look at an existing recording frame to see the "before" state:
  HF dataset amer224/territorial-bot-data → recordings/20260807-173018-492c50/frames/

STEP 2 (fastest reliable route — direct JS state): The game's inline JS has the
zoom core: `du.zoom(ft, fg, fh)` with `hs *= ft`. Try:
  a) page.evaluate("() => typeof du") — if it's a global, call
     page.evaluate("() => du.zoom(0.4, 0, 0)") repeatedly to zoom out, verify by
     measuring our blob % of screen (see SUCCESS below).
  b) If `du` is closure-private, intercept the wheel listener BEFORE the game
     loads with page.add_init_script that monkey-patches
     EventTarget.prototype.addEventListener to stash the first 'wheel' handler,
     then call that handler directly (no trusted-event problem):
        handler({type:'wheel', deltaY:-800, clientX:640, clientY:400,
                 preventDefault(){}, stopPropagation(){}})
     Repeat a few times until zoomed out.
  c) If the camera auto-follows and re-zooms, also patch/disable that behavior or
     re-zoom after recentering.

STEP 3 (fallback — trusted keys/buttons): If JS patch fails, find the game's
"Zoom In"/"Zoom Out" keybindings in its inline script (search "Zoom In","Zoom
Out" near the key-action table) and send trusted key events, or locate+click the
canvas zoom buttons by their drawn position.

STEP 4 (companion — recenter via leaderboard name): Implement/use
calibrate_from_leaderboard() + find_name_box() (already in src/bot/calibration.py)
to OCR your own row and CLICK your name → camera recenters on you at the new
zoom. This is the user's exact trick; combine it with the zoom-out.

SUCCESS CRITERION (verify with your own eyes on a fresh screenshot):
our territory blob is ≥1–5% of the screen AND ≥2 distinct enemy colors are
visible outside the leaderboard (use discover_enemies()/your vision). Then run
  python run_bot.py --record --games 5 --upload
and inspect a recorded frame mid-match to confirm enemies are visible.

───────────────────────────────────────────────────────────────────────────────

AFTER THE BLOCKER — the pipeline (already built, tested, needs real data)
- Recorder: run_bot.py --record --games N --upload → frames+clicks+colors+map
  OCR per match → HF amer224/territorial-bot-data/recordings/<session>/.
- Auto-labeler: scripts/label_real.py turns recordings+screenshots into 5-class
  per-pixel labels (water/neutral/me/enemy/ui) + click targets.
- Sim v8: src/sim/game6.py — REAL maps (weights/maps/*.npz from 26 real
  screenshots, 7 validated), 8–15 enemies, mixed-skill lobbies, kill tracking.
- Trainer: scripts/train_nn.py — stages collect/vision/clone/real/ppo/eval.
  v8 rewards: growth/2000 + kill×2 + win +5 (last survivor) − idle penalty;
  curriculum escalates skill AND enemy count. P100 fix built in (torch
  2.4.1+cu121 auto-install, _safe_device, FORCE_CPU=1).
- Kaggle: kaggle-push/kaggle_train_nn.ipynb → push to Kaggle as
  amerameryou/bot-train-nn, run on GPU. Auto-pulls recordings, labels them,
  vision+clone+real, PPO v8.
- Export: HF_TOKEN=... python3 scripts/export_hf.py → amer224/territorial-bot-nn.

PRIORITY ROADMAP (after zoom works)
1. Re-record 5–15 real matches (2–4 per map; switch maps between batches).
2. Set difficulty to Normal if automatable (user: "on easy bots don't attack
   each other"; Reset Scenario resets it — investigate the canvas control).
3. Fine-tune vision on real frames: label_real.py → train_nn.py real.
4. Kaggle v8 training run; pull log; verify alive-rate/win-rate numbers loudly.
5. Export best model to HF; update README with honest results + session table.
6. Long-term: boats/water, fix the 19 rejected maps, scale real data (~200k
   varied, NOT 2M procedural).

INVENTORY (workspace /home/user)
- bot-repo/ = the code (GitHub amerameryou1-blip/bot, public; README is the
  user's dashboard). key files: run_bot.py, src/bot/*, src/sim/game6.py,
  src/nn/*, scripts/*, tests/ (47 pass), weights/maps/ (7 real maps),
  weights/nn/ (last model: survives easy, never wins), QUALITY_PLAN.md,
  HANDOFF.md (full prior handoff).
- kaggle-push/ = Kaggle notebooks (main trainer + 4 workers, workers done).
- realdata/shots/ = the 26 real screenshots (mirrors HF screenshots/).
- datacheck/, uploads/ = user's custom-scenario setup screenshot (LOOK at it to
  understand the editor UI).
- territorial-bot/ = OLD deprecated project (has tune.py/tournament.py).

GAME FACTS (verified)
- Controls: double-click = claim land; Space = attack; W/S ±2% attack%; D/A
  ±0.5%; B = boat (3.125% tax); M = auto-attack; P = peace vote; H = hide UI.
  Rebindable key actions exist incl. "Zoom In"/"Zoom Out".
- Economy: interest 0.56s, income 5.6s, soft cap 100/px, hard cap 150/px, 7×
  boost first 107s, start 12px + 512 balance, defender 2:1 advantage.
- Custom Scenario editor: difficulty/colors partly canvas-rendered (clicking the
  DOM text did not cycle options — a vision agent should look at the editor
  screenshot and find the real control).
- Multiplayer lobby unreachable from datacenter IPs → Custom Scenario only.
- Leaderboard top-left; your name + color swatch always visible (OCR it).

WHAT "DONE" LOOKS LIKE (ship gates)
1. Zoom fixed: recorded frames show our territory ≥1–5% + ≥2 enemy colors.
2. ≥5 real matches recorded and uploaded to HF (usable frames + clicks).
3. Vision fine-tuned on real data; per-class gates: water ≥97%, me ≥90%,
   enemy ≥85%, UI ≥98% (on held-out real frames).
4. Policy: ≥30% last-survivor win vs 12-player mixed/hard sim lobby.
5. Real-game proof: bot wins 1 custom-scenario match AS LAST SURVIVOR (by
   elimination, not by area).
6. README updated with honest numbers; model exported to HF; workspace clean.

FINAL INSTRUCTIONS
Take your time. Reason carefully. Verify everything with tools and with your
eyes. The user is smart, engaged, and plays this game — treat them as a domain
expert, but investigate first before asking. You now have vision: USE IT. Look
at the game, look at the frames, look at the editor — that is the difference
between the previous agent failing and you succeeding.
```
