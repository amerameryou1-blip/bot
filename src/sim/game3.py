"""Vectorized headless territorial.io simulator (fast tournaments).

Mechanics (from the official tutorial + TerriEngine):
  - balance/troops: interest every 10 ticks + land income every 100 ticks,
    cap 150*land; expansion claims frontier land at 2 troops/pixel
  - attacks: adjacent players; the larger army captures the smaller's border
  - elimination at 0 pixels
"""
from __future__ import annotations

import numpy as np

from bot.economy import TroopTracker


class SimPlayer:
    __slots__ = ("name", "color", "troops", "alive", "rng", "bank")

    def __init__(self, name: str, color: int, seed: int):
        self.name = name
        self.color = color
        self.troops = TroopTracker(initial_troops=500)
        self.alive = True
        self.rng = np.random.default_rng(seed + color)
        self.bank = 0


def dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


class SimGame3:
    def __init__(self, h: int = 220, w: int = 300, n_bots: int = 3, seed: int = 1, max_ticks: int = 1400):
        self.h, self.w = h, w
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.max_ticks = max_ticks
        self.n_bots = n_bots

        self.world = np.zeros((h, w), dtype=np.int8)
        for _ in range(10):
            cy = int(self.rng.integers(0, h))
            cx = int(self.rng.integers(0, w))
            r = int(self.rng.integers(5, 14))
            yy, xx = np.ogrid[:h, :w]
            self.world[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = -1

        self.players: dict[int, SimPlayer] = {1: SimPlayer("OURS", 1, seed)}
        for i in range(n_bots):
            self.players[2 + i] = SimPlayer(f"bot{i}", 2 + i, seed + 100 + i)

        spots = np.argwhere(self.world == 0)
        self.rng.shuffle(spots)
        for i, pid in enumerate(sorted(self.players)):
            if i >= len(spots):
                break
            y, x = int(spots[i][0]), int(spots[i][1])
            yy, xx = np.ogrid[:h, :w]
            self.world[(yy - y) ** 2 + (xx - x) ** 2 < 4 ** 2] = pid

        self._pids = sorted(self.players)

    # -- observation for our planner --
    def state_for(self, pid: int = 1):
        from bot.state import Blob, FrameState
        labels = np.where(self.world == pid, 1, np.where(self.world > 0, 2, 0)).astype(np.int8)
        my_mask = labels == 1
        coords = np.argwhere(my_mask)
        me = None
        if len(coords):
            me = Blob("me", my_mask, len(coords), (float(coords[:, 0].mean()), float(coords[:, 1].mean())))
        enemies = []
        for opid in self.players:
            if opid == pid:
                continue
            m = self.world == opid
            if m.any():
                c = np.argwhere(m)
                enemies.append(Blob(f"e{opid}", m, len(c), (float(c[:, 0].mean()), float(c[:, 1].mean()))))
        from bot.vision import _find_frontiers
        frontiers = _find_frontiers(my_mask, labels == 0, max_samples=120) if me else np.zeros((0, 2), dtype=int)
        return FrameState(shape=(self.h, self.w), labels=labels, self_blob=me,
                          enemies=enemies, frontiers=frontiers)

    # -- one tick --
    def step(self, headings: dict[int, tuple[float, float]]):
        self.tick += 1
        for pid in self._pids:
            pl = self.players[pid]
            if not pl.alive:
                continue
            hx, hy = headings.get(pid, (0.0, 0.0))
            self._act(pl, hx, hy)
        self._resolve_attacks()
        self._eliminate()

    def _act(self, pl: SimPlayer, dx: float, dy: float):
        my = self.world == pl.color
        area = int(my.sum())
        if area == 0:
            pl.alive = False
            return
        pl.troops.update(area)
        if abs(dx) < 0.25 and abs(dy) < 0.25:
            pl.bank += 1
            return

        neutral = (self.world == 0) & dilate4(my)
        if not neutral.any():
            return
        idx = np.argwhere(neutral)
        cy, cx = float(np.argwhere(my)[:, 0].mean()), float(np.argwhere(my)[:, 1].mean())
        vecs = idx.astype(float) - np.array([cy, cx])
        dots = vecs[:, 0] * dy + vecs[:, 1] * dx
        order = np.argsort(-dots)
        cap = min(90, max(10, area // 60))
        # vectorized: claim as many as troops allow
        affordable = int(pl.troops.troops // 2)
        n_claim = min(len(order), cap, affordable)
        if n_claim <= 0:
            return
        claimed_idx = order[:n_claim]
        self.world[idx[claimed_idx, 0], idx[claimed_idx, 1]] = pl.color
        pl.troops.claim(n_claim)

    def _resolve_attacks(self):
        for a in self._pids:
            for b in self._pids:
                if b <= a:
                    continue
                pa, pb = self.players[a], self.players[b]
                if not (pa.alive and pb.alive):
                    continue
                ma, mb = self.world == a, self.world == b
                if not (dilate4(ma) & mb).any():
                    continue
                aa, ab = int(ma.sum()), int(mb.sum())
                if aa > ab * 1.3:
                    border = mb & dilate4(ma)
                    self.world[border] = a
                    pb.troops.troops = max(0.0, pb.troops.troops - border.sum() * 2)
                elif ab > aa * 1.3:
                    border = ma & dilate4(mb)
                    self.world[border] = b
                    pa.troops.troops = max(0.0, pa.troops.troops - border.sum() * 2)

    def _eliminate(self):
        for pid in self._pids:
            if (self.world == pid).sum() == 0:
                self.players[pid].alive = False

    # -- full match --
    def run_match(self, our_heading_fn):
        our_max = 0
        report = []
        while self.tick < self.max_ticks:
            headings: dict[int, tuple[float, float]] = {}
            if self.players[1].alive:
                state = self.state_for(1)
                if state.self_blob:
                    plan = our_heading_fn(state)
                    headings[1] = (plan.dx, plan.dy)
            for pid in self._pids:
                if pid == 1 or not self.players[pid].alive:
                    continue
                headings[pid] = self._bot_heading(pid)
            self.step(headings)
            a = int((self.world == 1).sum())
            our_max = max(our_max, a)
            if self.tick % 350 == 0:
                report.append({"tick": self.tick, "our_area": a})
            alive = [pid for pid in self._pids if self.players[pid].alive]
            if len(alive) <= 1:
                break
        alive = [pid for pid in self._pids if self.players[pid].alive]
        winner = alive[0] if len(alive) == 1 else (1 if self.players[1].alive else (alive[0] if alive else 0))
        return {"winner": winner, "our_max_area": our_max,
                "our_final_area": int((self.world == 1).sum()),
                "ticks": self.tick, "alive": alive, "report": report}

    def _bot_heading(self, pid: int) -> tuple[float, float]:
        pl = self.players[pid]
        my = self.world == pid
        if not my.any():
            return (0.0, 0.0)
        neutral = (self.world == 0) & dilate4(my)
        if not neutral.any():
            return (0.0, 0.0)
        idx = np.argwhere(neutral)
        cy, cx = float(np.argwhere(my)[:, 0].mean()), float(np.argwhere(my)[:, 1].mean())
        my_area = int(my.sum())
        # score each frontier claim: distance + fear of bigger enemies
        big = np.zeros_like(self.world, dtype=bool)
        for opid in self._pids:
            if opid == pid or not self.players[opid].alive:
                continue
            om = self.world == opid
            if int(om.sum()) > my_area * 1.2:
                big |= dilate4(om)
        dist = np.hypot(idx[:, 0] - cy, idx[:, 1] - cx)
        fear = np.zeros(len(idx), dtype=float)
        if big.any():
            fear = 300.0 / (1.0 + dist)  # rough: frontier near big enemy is scary
        # approximate fear by distance to big mask
        bcoords = np.argwhere(big)
        if len(bcoords):
            d2 = ((idx[:, None, :] - bcoords[None, :, :]) ** 2).sum(axis=2)
            mind = d2.min(axis=1)
            fear = 300.0 / (1.0 + mind)
        score = dist - fear
        best = int(np.argmax(score))
        y, x = idx[best]
        v = np.array([float(y - cy), float(x - cx)])
        mag = float(np.hypot(*v))
        if mag < 1e-6:
            return (0.0, 0.0)
        return (v[1] / mag, v[0] / mag)
