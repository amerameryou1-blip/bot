"""Fast headless territorial.io simulator with real attack mechanics.

Model (based on TerriEngine):
  - troops: interest every 10 ticks, +land every 100 ticks, cap 150*land
  - expansion: claim frontier neutral pixels at 2 troops each, up to a
    per-tick cap proportional to troops
  - attack: when two players are adjacent, the attacker with more local
    troops captures border pixels; a player with 0 area is eliminated

Optimized with numpy where possible so a full match runs in seconds.
"""
from __future__ import annotations

import numpy as np

from ..bot.economy import TroopTracker


class SimPlayer:
    __slots__ = ("name", "color", "troops", "alive", "heading", "rng")

    def __init__(self, name: str, color: int, seed: int):
        self.name = name
        self.color = color
        self.troops = TroopTracker(initial_troops=400)
        self.alive = True
        self.heading = np.array([1.0, 0.0])
        self.rng = np.random.default_rng(seed + color)


def dilate(mask: np.ndarray) -> np.ndarray:
    """4-neighbour dilation."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


class SimGame2:
    def __init__(self, h: int = 220, w: int = 300, n_bots: int = 3, seed: int = 1, max_ticks: int = 1500):
        self.h, self.w = h, w
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.max_ticks = max_ticks

        # 0 = neutral land, -1 = water
        self.world = np.zeros((h, w), dtype=np.int8)
        for _ in range(10):
            cy = self.rng.integers(0, h)
            cx = self.rng.integers(0, w)
            r = self.rng.integers(5, 14)
            yy, xx = np.ogrid[:h, :w]
            self.world[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = -1

        self.players: dict[int, SimPlayer] = {1: SimPlayer("OURS", 1, seed)}
        for i in range(n_bots):
            self.players[2 + i] = SimPlayer(f"bot{i}", 2 + i, seed + 100 + i)

        spots = np.argwhere(self.world == 0)
        self.rng.shuffle(spots)
        for i, pid in enumerate(sorted(self.players)):
            y, x = spots[i]
            yy, xx = np.ogrid[:h, :w]
            self.world[(yy - y) ** 2 + (xx - x) ** 2 < 4 ** 2] = pid

    def state_for(self, pid: int = 1):
        """FrameState-style observation for our planner."""
        from ..bot.state import Blob, FrameState
        labels = np.where(self.world == pid, 1, np.where(self.world > 0, 2, 0)).astype(np.int8)
        my_mask = labels == 1
        coords = np.argwhere(my_mask)
        me = None
        if len(coords):
            me = Blob("me", my_mask, len(coords),
                      (float(coords[:, 0].mean()), float(coords[:, 1].mean())))
        enemies = []
        for opid in self.players:
            if opid == pid:
                continue
            m = self.world == opid
            if m.any():
                c = np.argwhere(m)
                enemies.append(Blob(f"e{opid}", m, len(c),
                                    (float(c[:, 0].mean()), float(c[:, 1].mean()))))
        from ..bot.vision import _find_frontiers
        frontiers = _find_frontiers(my_mask, labels == 0, max_samples=120) if me else np.zeros((0, 2), dtype=int)
        return FrameState(shape=(self.h, self.w), labels=labels, self_blob=me,
                          enemies=enemies, frontiers=frontiers)

    def step(self, headings: dict[int, tuple[float, float]]):
        self.tick += 1
        for pid, player in self.players.items():
            if not player.alive:
                continue
            hx, hy = headings.get(pid, (0.0, 0.0))
            self._act(player, hx, hy)
        self._resolve_attacks()
        self._eliminate()

    def _act(self, player: SimPlayer, dx: float, dy: float):
        my_mask = self.world == player.color
        area = int(my_mask.sum())
        if area == 0:
            player.alive = False
            return
        player.troops.update(area)
        if abs(dx) < 0.25 and abs(dy) < 0.25:
            return  # banking

        # neutral cells adjacent to my territory (frontier to claim)
        neutral = (self.world == 0) & dilate(my_mask)
        if not neutral.any():
            return
        idx = np.argwhere(neutral)
        cy, cx = np.argwhere(my_mask).mean(axis=0)
        vecs = idx.astype(float) - np.array([cy, cx])
        dots = vecs[:, 0] * dy + vecs[:, 1] * dx
        order = np.argsort(-dots)
        cap = min(80, max(8, int(area // 50)))  # claim rate scales with area
        claimed = 0
        for j in order:
            if claimed >= cap:
                break
            if not player.troops.can_claim(1):
                break
            y, x = int(idx[j][0]), int(idx[j][1])
            player.troops.claim(1)
            self.world[y, x] = player.color
            claimed += 1

    def _resolve_attacks(self):
        """Adjacent players: the bigger one captures the smaller's border."""
        pids = list(self.players)
        for a in pids:
            if not self.players[a].alive:
                continue
            for b in pids:
                if b <= a or not self.players[b].alive:
                    continue
                ma = self.world == a
                mb = self.world == b
                if not (dilate(ma) & mb).any():
                    continue
                aa, ab = int(ma.sum()), int(mb.sum())
                if aa > ab * 1.25:
                    # a captures b's border pixels
                    border_b = mb & dilate(ma)
                    self.world[border_b] = a
                    self.players[b].troops.troops = max(0, self.players[b].troops.troops - border_b.sum() * 2)
                elif ab > aa * 1.25:
                    border_a = ma & dilate(mb)
                    self.world[border_a] = b
                    self.players[a].troops.troops = max(0, self.players[a].troops.troops - border_a.sum() * 2)

    def _eliminate(self):
        for pid, player in self.players.items():
            if (self.world == pid).sum() == 0:
                player.alive = False

    def run_match(self, our_heading_fn, report_every=300):
        our_max = 0
        report = []
        while self.tick < self.max_ticks:
            headings: dict[int, tuple[float, float]] = {}
            if self.players[1].alive:
                state = self.state_for(1)
                if state.self_blob:
                    plan = our_heading_fn(state)
                    headings[1] = (plan.dx, plan.dy)
            for pid, player in self.players.items():
                if pid == 1 or not player.alive:
                    continue
                headings[pid] = self._bot_heading(pid)
            self.step(headings)
            a = int((self.world == 1).sum())
            our_max = max(our_max, a)
            if self.tick % report_every == 0:
                report.append({"tick": self.tick, "our_area": a})
            alive = [pid for pid, pl in self.players.items() if pl.alive]
            if len(alive) <= 1:
                break
        alive = [pid for pid, pl in self.players.items() if pl.alive]
        winner = alive[0] if len(alive) == 1 else (1 if self.players[1].alive else alive[0] if alive else 0)
        return {"winner": winner, "our_max_area": our_max,
                "our_final_area": int((self.world == 1).sum()),
                "ticks": self.tick, "alive": alive, "report": report}

    def _bot_heading(self, pid: int) -> tuple[float, float]:
        player = self.players[pid]
        my_mask = self.world == pid
        if not my_mask.any():
            return (0.0, 0.0)
        cy, cx = np.argwhere(my_mask).mean(axis=0)
        neutral = (self.world == 0) & dilate(my_mask)
        if not neutral.any():
            return (0.0, 0.0)
        idx = np.argwhere(neutral)
        # prefer neutral away from bigger enemies, toward open space
        best = None
        best_score = -1e9
        sample = idx[::3]
        for (y, x) in sample:
            d = float(np.hypot(y - cy, x - cx))
            score = d
            for opid, opl in self.players.items():
                if opid == pid or not opl.alive:
                    continue
                om = self.world == opid
                if om.sum() < my_mask.sum() * 1.2:
                    continue
                oc = np.argwhere(om).mean(axis=0)
                ed = float(np.hypot(y - oc[0], x - oc[1]))
                if ed < 45:
                    score -= 300
            if score > best_score:
                best_score = score
                best = (float(y - cy), float(x - cx))
        if best is None:
            return (0.0, 0.0)
        mag = max(float(np.hypot(*best)), 1e-6)
        return (best[1] / mag, best[0] / mag)
