# Territorial.io — ATTACK META (researched 2026-08-07)

Synthesized from community strategy guides / Reddit / gameplay sites.
This is the strategy the neural net is trained to learn.

## Core mechanics (for attack decisions)
- **Defender has ~2:1 advantage** → you need roughly 2× the defender's balance
  to conquer efficiently. Less wastes troops; more is unnecessary.
- **Attack tax: 1.17% of your balance per attack** → don't spam attacks.
- **Attack rhythm**: one attack per 5.6s income cycle, timed to land near the
  income tick (end of cycle) to offset cost with newly conquered land income.
- **Economy**: soft cap 100 troops/px (bar turns RED = max interest), hard cap
  150/px (interest stops). Save to red before sustained attacking.
- Boats: 3.125% tax to build, 1% troops to claim distant islands.

## The phases (win = LAST SURVIVOR, not biggest area)
1. **Opening (~first minutes)**: take ALL free (neutral) land first. Expand at
   ~20-30% per cycle (one expansion per 5.6s tick). Do NOT attack others yet —
   attackers lose more than defenders early.
2. **Bot phase (free land gone)**: attack bots with ~10% (5-12%). KEY TIMING:
   **attack a bot right AFTER it attacks** — bots full-send and are left with
   ~0 troops, making them cheap to conquer. Always attack the weakest /
   lowest-troop bot first. Never use high % on bots.
3. **Mid game**: save until red interest (~100/px). Attack the weakest /
   least-dense neighbor at ~10-11% per cycle. Attack only when you have ~2:1
   advantage. Prefer easy land over attacking the biggest player.
4. **Late game (crown dynamics)**: if you're the leader, hold truces, defend,
   don't attack first (they'll unload on you). If not the leader, never let the
   leader reach ~45% of the map — gang up / attack the weakest significant
   player to slow them. When you have ~50%+ land at red interest, attack the
   largest bordering player at 10% every cycle until you win.

## Bots (what to exploit)
- Bots full-send when attacking → immediately after, they have almost no troops.
- Bots expand inefficiently and leave borders under-defended.
- Always kill bots before players.

## Implications for the bot's brain
- Attack decisions need: enemy BALANCE (not just area), enemy "just attacked"
  status, my density (red?), phase (free land? bots? players?), 2:1 check.
- The neural net should learn: WHEN to attack (phase + enemy weakness + red),
  WHERE (weakest bordering enemy, low density), HOW MUCH (10% bots / 10-11% players).
