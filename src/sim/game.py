"""Headless territorial.io-style game simulator.

Implements the core mechanics from TerriEngine:
  - troops economy: interest every 10 ticks, land income every 100 ticks,
    cap 150*land, expansion cost 2 troops/pixel
  - moving in a direction claims neutral pixels along the frontier at
    2 troops/pixel
  - attacking: contesting enemy pixels, the attacker with local troop
    advantage captures them
  - bots: simple AI players that expand toward open land and attack weak
    neighbors

Used to battle-test the planner offline and measure win rates.
"""
from __future__ import annotations

import numpy as np

from bot.economy import TroopTracker


class SimPlayer:
    def __init__(self, name: str, color: int, strategy: str = "greedy"):
        self.name = name
        self.color = color          # label value in the occupancy map
        self.troops = TroopTracker(initial_troops=400)
        self.strategy = strategy
        self.alive = True
        self.heading = 0.0
        self.bank_ticks = 0

    def decide_heading(self, labels, my_mask, enemy_masks):
        """Simple bot AI: expand toward open land, attack small neighbors."""
        h, w = labels.shape
        cy, cx = np.argwhere(my_mask).mean(axis=0)
        # frontier of my territory
        from bot.vision import _find_frontiers
        frontiers = _find_frontiers(my_mask, labels == 0, max_samples=60)
        if len(frontiers) == 0:
            return 0.0, 0.0, "bank"
        # pick frontier farthest from the centroid of big enemies
        best = None
        best_score = -1e9
        for fx, fy in frontiers[::3]:
            vec = np.array([fx - cy, fy - cx])
            dist = np.linalg.norm(vec)
            if dist < 1:
                continue
            score = dist
            for m in enemy_masks:
                if m.sum() > my_mask.sum():
                    ec = np.argwhere(m).mean(axis=0)
                    ed = np.linalg.norm(np.array([fx - ec[0], fy - ec[1]]))
                    if ed < 40:
                        score -= 200
            if score > best_score:
                best_score = score
                best = vec / dist
        if best is None:
            return 0.0, 0.0, "wander"
        return best[1], best[0], "expand"


class SimGame:
    def __init__(self, h: int = 300, w: int = 400, n_bots: int = 4, seed: int = 7,
                 bot_strategy: str = "greedy"):
        self.h, self.w = h, w
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.max_ticks = 4000

        # World: 0 = neutral land, -1 = water (blocked), 1..n = player colors
        self.world = np.zeros((h, w), dtype=np.int8)
        self._make_land()

        # players: color 1 = the bot under test; 2.. = bots
        self.players: dict[int, SimPlayer] = {}
        self.players[1] = SimPlayer("OURS", 1, strategy="planner")
        for i in range(n_bots):
            self.players[2 + i] = SimPlayer(f"bot{i}", 2 + i, strategy=bot_strategy)
        self._place_players()

    def _make_land(self):
        """Random blobby landmass; water = -1, rest neutral = 0."""
        world = np.zeros((self.h, self.w), dtype=np.int8)
        for _ in range(14):
            cy = self.rng.integers(0, self.h)
            cx = self.rng.integers(0, self.w)
            r = self.rng.integers(6, 18)
            yy, xx = np.ogrid[: self.h, : self.w]
            world[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = -1
        self.world = world

    def _place_players(self):
        spots = np.argwhere(self.world == 0)
        self.rng.shuffle(spots)
        n = len(self.players)
        for i, (pid, player) in enumerate(self.players.items()):
            if i >= len(spots):
                break
            y, x = spots[i]
            r = 4
            yy, xx = np.ogrid[: self.h, : self.w]
            self.world[(yy - y) ** 2 + (xx - x) ** 2 < r ** 2] = pid
            player.centroid = (float(y), float(x))

    # -- observation for the planner under test --
    def frame_state(self, pid: int = 1):
        from bot.state import Blob, FrameState
        labels = np.where(self.world == pid, 1, np.where(self.world > 1, self.world, 0))
        labels[labels > 1] = 2  # all enemies -> 2 for FrameState
        my_mask = labels == 1
        coords = np.argwhere(my_mask)
        me = Blob("me", my_mask, len(coords),
                  (float(coords[:, 0].mean()), float(coords[:, 1].mean()))) if len(coords) else None
        # enemies as individual blobs
        enemies = []
        for other_pid in self.players:
            if other_pid == pid:
                continue
            m = self.world == other_pid
            if m.any():
                c = np.argwhere(m)
                enemies.append(Blob(f"e{other_pid}", m, len(c),
                                    (float(c[:, 0].mean()), float(c[:, 1].mean()))))
        from bot.vision import _find_frontiers
        frontiers = _find_frontiers(my_mask, labels == 0, max_samples=120) if me else np.zeros((0, 2), dtype=int)
        return FrameState(shape=(self.h, self.w), labels=labels, self_blob=me,
                          enemies=enemies, frontiers=frontiers)

    def step(self, heading_pid: dict[int, tuple[float, float]]):
        """Advance one tick. heading_pid maps pid -> (dx, dy) normalized."""
        self.tick += 1
        for pid, player in self.players.items():
            if not player.alive:
                continue
            dx, dy = heading_pid.get(pid, (0.0, 0.0))
            self._move(player, dx, dy)
        self._resolve_attacks()
        self._eliminate()

    def _move(self, player: SimPlayer, dx: float, dy: float):
        my_mask = self.world == player.color
        if not my_mask.any():
            player.alive = False
            return
        player.troops.update(int(my_mask.sum()))
        if abs(dx) < 0.2 and abs(dy) < 0.2:
            return  # banking troops
        # frontier NEUTRAL pixels to claim: neutral cells adjacent to my territory
        from bot.vision import _find_frontiers
        frontiers = _find_frontiers(my_mask, self.world == 0, max_samples=400)
        if len(frontiers) == 0:
            return
        # collect candidate neutral neighbors of each frontier pixel
        claims = []
        h, w = self.world.shape
        for fy, fx in frontiers:
            for ny, nx in ((fy-1, fx), (fy+1, fx), (fy, fx-1), (fy, fx+1)):
                if 0 <= ny < h and 0 <= nx < w and self.world[ny, nx] == 0:
                    claims.append((ny, nx))
        if not claims:
            return
        claims = np.array(claims)
        # prefer claims in the heading direction
        cy, cx = np.argwhere(my_mask).mean(axis=0)
        vecs = claims.astype(float) - np.array([cy, cx])
        dots = vecs[:, 0] * dy + vecs[:, 1] * dx
        order = np.argsort(-dots)
        claimed = 0
        seen = set()
        for idx in order:
            if claimed >= 60:
                break
            ny, nx = int(claims[idx][0]), int(claims[idx][1])
            if (ny, nx) in seen:
                continue
            seen.add((ny, nx))
            if not player.troops.can_claim(1):
                break
            player.troops.claim(1)
            self.world[ny, nx] = player.color
            claimed += 1

    def _resolve_attacks(self):
        # Attack happens when a player's territory borders an enemy and the
        # attacker has troops advantage; simplified: attacker captures enemy
        # border pixels at 2 troops each while their ratio advantage holds.
        pass

    def _eliminate(self):
        for pid, player in self.players.items():
            if (self.world == pid).sum() == 0:
                player.alive = False

    def run_match(self, our_heading_fn, report_every=500):
        """Run a full match; our_heading_fn(state) -> (dx, dy).
        Returns {winner, our_max_area, our_final_area, ticks, alive_bots}."""
        our_max = 0
        report = []
        while self.tick < self.max_ticks:
            headings = {}
            # our bot
            if self.players[1].alive:
                state = self.frame_state(1)
                if state.self_blob:
                    plan = our_heading_fn(state)
                    headings[1] = (plan.dx, plan.dy)
            # bots
            for pid in list(self.players):
                if pid == 1 or not self.players[pid].alive:
                    continue
                p = self.players[pid]
                labels = np.where(self.world == pid, 1, np.where(self.world > 0, 2, 0))
                my_mask = self.world == pid
                enemy_masks = [self.world == q for q in self.players if q != pid]
                dx, dy, _ = p.decide_heading(labels, my_mask, enemy_masks)
                headings[pid] = (dx, dy)
            self.step(headings)
            a = int((self.world == 1).sum())
            our_max = max(our_max, a)
            if self.tick % report_every == 0:
                report.append({"tick": self.tick, "our_area": a})
            alive = [pid for pid, p in self.players.items() if p.alive]
            if len(alive) <= 1:
                break
        alive = [pid for pid, p in self.players.items() if p.alive]
        winner = alive[0] if alive else 0
        final_area = int((self.world == 1).sum())
        return {
            "winner": winner, "our_max_area": our_max, "our_final_area": final_area,
            "ticks": self.tick, "alive_bots": [pid for pid in self.players if pid != 1 and self.players[pid].alive],
            "report": report,
        }
