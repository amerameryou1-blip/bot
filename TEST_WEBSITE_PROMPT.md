# 🧪 TEST PROMPT — "Arena: Territory Conquest" (single-file web game)

> Copy EVERYTHING in the code fence below and paste it as one message to the
> next agent. It is a **skill test**: a deep-thinking build of a complex
> self-contained website. If the agent thinks hard, produces a polished
> playable result, and verifies it with tools + vision, then it is ready for
> the real project (the territorial.io bot). If it rushes or half-builds,
> don't continue with it.

---

```
ROLE
You are a senior creative engineer + game developer. Build, from scratch, a
single self-contained web application: a polished real-time territory
conquest game playable in a browser. This is a TEST of your ability to think
deeply about architecture, build something complex and beautiful, and verify
it works. Take your time. Quality over speed. A rushed half-game is a FAIL.

THE PRODUCT — "ARENA: TERRITORY CONQUEST"
A single-page canvas game where the player expands territory, gathers
resources, and fights AI-controlled rivals. The last player standing wins.

Core features (all required):
1. PROCEDURAL MAP — a grid world with LAND, WATER (impassable lakes), and a
   few MOUNTAIN cells (impassable). Generate it from a seed so maps vary;
   show the seed and allow re-rolling. Coastlines should look organic
   (noise-based), not like rectangles.
2. TERRITORY — the player starts with a small claimed area. Click (or click +
   drag) on adjacent unclaimed land to expand, at a cost of troops. Claimed
   cells are drawn in the player's color with a subtle border glow.
3. ECONOMY — a troop/balance resource that ticks up over time (income rate
   should scale with territory size, with diminishing returns). Display it.
   Expansion costs troops; attacking costs troops.
4. AI BOTS — 3-6 bots with distinct colors that expand on their own and
   attack the weakest adjacent opponent when strong enough. They should feel
   alive: some aggressive, some cautious. Bots must attack each other too —
   it's a free-for-all.
5. COMBAT — attacking an adjacent enemy cell transfers troops; stronger force
   wins. When a bot loses ALL its cells it is ELIMINATED (name shown in a
   kill feed). Game ends when one player remains → WIN/LOSE screen with the
   winner's name, and a restart button.
6. CAMERA — the map is bigger than the screen: support drag-to-pan and
   scroll-to-zoom (this is important — test it carefully, wheel zoom must
   work reliably).
7. HUD — top-left leaderboard (rank, name, cells, alive/dead), top-right
   economy display, bottom hint bar, pause/restart buttons, seed display.
8. POLISH — smooth 60fps, subtle animations (expand pulse, attack flashes,
   elimination explosion), particle trails on attacks, clean modern UI
   (dark theme), hover cursor feedback. No sound required.
9. LAST SURVIVOR — the win condition is being the last one standing, not the
   biggest territory. If you die, show "you placed N of M".

HARD TECHNICAL CONSTRAINTS (violating these = FAIL)
- ONE self-contained file (index.html) with ALL CSS and JS inline.
- ZERO external dependencies: no CDNs, no npm, no webfonts, no images —
  everything must come from the file itself (inline styles, canvas drawing,
  system fonts, data URIs). It must work fully offline and inside a
  sandboxed iframe with no network access.
- The game loop should run on requestAnimationFrame with a fixed-timestep
  update (never let real time drift the sim) and delta-time interpolation.
- Use canvas 2D for the world; DOM overlays only for HUD.
- No console errors. Ever.

MANDATORY PROCESS (this is the point of the test — follow it exactly)
1. THINK FIRST. Before writing any code, write a short architecture plan as
   text: data model (map grid, players, economy), the game state machine,
   how update/render are separated, how combat resolves, how AI decides, how
   pan/zoom is implemented, how win/lose is detected. Reason about edge cases
   (clicking water, clicking your own cell, attack when balance too low,
   last two players, pause during zoom). Write the plan down BEFORE coding.
2. Build in layers, verifying each: (a) blank canvas + 60fps loop,
   (b) map generation + rendering, (c) expand + economy, (d) bots + combat +
   elimination, (e) camera pan/zoom, (f) HUD + win/lose + polish.
3. RUN AND VERIFY WITH TOOLS — don't just write it: serve it (bind to
   0.0.0.0, use relative URLs), open it headlessly with Playwright, and
   ACTUALLY TEST: click to expand, verify economy ticks, verify bots move,
   verify an elimination happens, verify zoom/pan works, check console errors
   = none, screenshot mid-game and LOOK at it with your vision. Fix anything
   you see that's wrong or ugly. Loop until it's genuinely good.
4. Only then declare it done, and give the user a 3-5 line summary of the
   architecture and how to play.

ACCEPTANCE CRITERIA (the user will check these with their own eyes)
- Opens in the browser and the map renders immediately.
- Clicking land expands territory; economy numbers change.
- Bots visibly expand and fight (kill feed shows eliminations).
- Scroll zooms, drag pans — reliable.
- A match can be won and lost; the win screen names the survivor.
- Looks polished, not like a prototype: good colors, animation, layout.
- No console errors; runs smoothly.

FINAL INSTRUCTION
Reason carefully about every system before coding it. Verify everything with
tools and with your eyes. A deep, complete, beautiful result is the goal —
this test decides whether you take on the bigger project next.
```
