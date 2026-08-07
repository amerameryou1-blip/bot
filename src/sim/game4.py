"""Click-based headless simulator — mirrors the REAL mechanics.

Per the mechanics doc:
  - expand: click a neutral frontier pixel -> claim it (2 troops/pixel)
  - attack: click enemy border with slider % -> spend balance*pct; defenders
    fight at 2:1 advantage; capture happens if attacker force > 2× defender
  - economy: interest tick 0.56s, income tick 5.6s, soft/hard limits
  - win: 100% land, or last alive
Used to battle-test the ClickPlanner offline against bots.
"""
from __future__ import annotations

import numpy as np

from bot.economy import HARD_LIMIT, NEUTRAL_PIXEL_COST, SOFT_LIMIT, TICK_S, TroopTracker
from bot.vision import _adjacent_mask, find_attack_targets, find_expand_targets


class ClickPlayer:
    __slots__ = ("name", "color", "troops", "alive", "rng", "_pending_attack")

    def __init__(self, name: str, color: int, seed: int):
        self.name = name
        self.color = color
        self.troops = TroopTracker(balance=512.0, land=12)
        self.alive = True
        self.rng = np.random.default_rng(seed + color)
        self._pending_attack = None


class ClickSim:
    def __init__(self, h: int = 220, w: int = 300, n_bots: int = 3, seed: int = 1,
                 max_ticks: int = 1600, clicks_per_tick: int = 3):
        self.h, self.w = h, w
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.max_ticks = max_ticks
        self.clicks_per_tick = clicks_per_tick

        self.world = np.zeros((h, w), dtype=np.int16)
        for _ in range(10):
            cy = int(self.rng.integers(0, h))
            cx = int(self.rng.integers(0, w))
            r = int(self.rng.integers(5, 14))
            yy, xx = np.ogrid[:h, :w]
            self.world[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = -1

        self.players: dict[int, ClickPlayer] = {1: ClickPlayer("OURS", 1, seed)}
        for i in range(n_bots):
            self.players[2 + i] = ClickPlayer(f"bot{i}", 2 + i, seed + 100 + i)
        self._pids = sorted(self.players)

        spots = np.argwhere(self.world == 0)
        self.rng.shuffle(spots)
        for i, pid in enumerate(self._pids):
            if i >= len(spots):
                break
            y, x = int(spots[i][0]), int(spots[i][1])
            yy, xx = np.ogrid[:h, :w]
            self.world[(yy - y) ** 2 + (xx - x) ** 2 < 4 ** 2] = pid

    # -- observation -------------------------------------------------------

    def state_for(self, pid: int = 1):
        from bot.state import Blob, FrameState
        labels = np.where(self.world == pid, 1, np.where(self.world > 0, 2, 0)).astype(np.int8)
        my_mask = labels == 1
        coords = np.argwhere(my_mask)
        me = None
        if len(coords):
            me = Blob("me", my_mask, len(coords), (float(coords[:, 0].mean()), float(coords[:, 1].mean())))
        enemies = []
        for opid in self._pids:
            if opid == pid:
                continue
            m = self.world == opid
            if m.any():
                c = np.argwhere(m)
                enemies.append(Blob(f"e{opid}", m, len(c), (float(c[:, 0].mean()), float(c[:, 1].mean()))))
        enemy_mask = labels >= 2
        expand = find_expand_targets(my_mask, labels == 0, max_samples=160) if me else np.zeros((0, 2), dtype=int)
        attack = find_attack_targets(my_mask, enemy_mask, max_samples=120) if me else np.zeros((0, 2), dtype=int)
        return FrameState(shape=(self.h, self.w), labels=labels, self_blob=me,
                          enemies=enemies, frontiers=np.zeros((0, 2), dtype=int),
                          expand_targets=expand, attack_targets=attack)

    # -- stepping ------------------------------------------------------------

    def step(self, actions: dict[int, list[tuple[str, int, int, float]]]):
        """actions: pid -> list of (kind, y, x, pct) — 'expand' or 'attack'."""
        self.tick += 1
        for pid in self._pids:
            pl = self.players[pid]
            if not pl.alive:
                continue
            my = self.world == pid
            area = int(my.sum())
            if area == 0:
                pl.alive = False
                continue
            pl.troops.update(area)
            for kind, y, x, pct in actions.get(pid, []):
                self._apply_click(pl, kind, y, x, pct)
        self._resolve_attacks()
        self._eliminate()

    def _apply_click(self, pl: ClickPlayer, kind: str, y: int, x: int, pct: float):
        if not (0 <= y < self.h and 0 <= x < self.w):
            return
        if kind == "expand":
            if self.world[y, x] == 0 and pl.troops.can_afford_expansion(1):
                pl.troops.spend_expansion(1)
                self.world[y, x] = pl.color
        elif kind == "attack":
            target = int(self.world[y, x])
            if target <= 0 or target == pl.color or target not in self.players:
                return
            # queue an attack: stored on the player for resolution
            budget = pl.troops.attack_budget(pct)
            pl._pending_attack = (target, budget, y, x)  # type: ignore[attr-defined]

    def _resolve_attacks(self):
        for pid in self._pids:
            pl = self.players[pid]
            if not pl.alive:
                continue
            pending = getattr(pl, "_pending_attack", None)
            if not pending:
                continue
            target, budget, y, x = pending
            del pl._pending_attack  # type: ignore[attr-defined]
            defender = self.players.get(target)
            if not defender or not defender.alive:
                continue
            # defender fights at 2:1 → effective defense = 2 × defender balance
            defense = 2.0 * defender.troops.balance
            if budget > defense:
                over = budget - defense
                defender.troops.balance = max(0.0, defender.troops.balance - budget * 0.6)
                # capture border pixels of the defender adjacent to attacker
                border = (self.world == target) & _adjacent_mask(self.world == pid)
                capture_n = min(int(border.sum()), int(over // (2 * NEUTRAL_PIXEL_COST)) + 1)
                if capture_n > 0:
                    idx = np.argwhere(border)
                    rng = pl.rng
                    pick = rng.choice(len(idx), size=min(capture_n, len(idx)), replace=False)
                    for i in pick:
                        yy, xx = idx[i]
                        self.world[yy, xx] = pid
                        pl.troops.land += 1

    def _eliminate(self):
        for pid in self._pids:
            if (self.world == pid).sum() == 0:
                self.players[pid].alive = False

    # -- full match -----------------------------------------------------------

    def run_match(self, our_click_fn, report_every=300):
        """our_click_fn(state) -> ClickAction. Returns result dict."""
        our_max = 0
        report = []
        while self.tick < self.max_ticks:
            actions: dict[int, list] = {}
            if self.players[1].alive:
                state = self.state_for(1)
                if state.self_blob:
                    # feed enemy balances so the planner can target exhausted ones
                    if hasattr(our_click_fn, "planner"):
                        balances = {f"e{pid}": self.players[pid].troops.balance
                                    for pid in self._pids if pid != 1}
                        our_click_fn.planner.set_enemy_balances(balances)
                    act = our_click_fn(state)
                    actions[1] = self._clicks_for(act, self.clicks_per_tick)
            for pid in self._pids:
                if pid == 1 or not self.players[pid].alive:
                    continue
                actions[pid] = self._bot_clicks(pid)
            self.step(actions)
            a = int((self.world == 1).sum())
            our_max = max(our_max, a)
            if self.tick % report_every == 0:
                report.append({"tick": self.tick, "our_area": a})
            alive = [pid for pid in self._pids if self.players[pid].alive]
            if len(alive) <= 1:
                break
        alive = [pid for pid in self._pids if self.players[pid].alive]
        winner = alive[0] if len(alive) == 1 else (1 if self.players[1].alive else (alive[0] if alive else 0))
        return {"winner": winner, "our_max_area": our_max,
                "our_final_area": int((self.world == 1).sum()),
                "ticks": self.tick, "alive": alive, "report": report}

    def _clicks_for(self, act, n):
        """Convert one ClickAction into up to n DISTINCT sim clicks.

        One planner decision = one click budget burst: expand claims several
        distinct frontier pixels near the chosen target (the real bot clicks
        2-3×/s, several per conquest tick).
        """
        clicks = []
        if act.kind == "expand":
            st = self.state_for(1)
            targets = st.expand_targets
            if len(targets):
                ty, tx = int(act.y), int(act.x)
                d = (targets[:, 0] - ty) ** 2 + (targets[:, 1] - tx) ** 2
                order = np.argsort(d)[:n]
                for i in order:
                    y, x = int(targets[i][0]), int(targets[i][1])
                    clicks.append(("expand", y, x, 0.0))
        elif act.kind == "attack":
            clicks.append(("attack", int(act.y), int(act.x), act.pct))
        return clicks

    def _bot_clicks(self, pid: int) -> list:
        pl = self.players[pid]
        st = self.state_for(pid)
        if not st.self_blob:
            return []
        clicks = []
        my_area = st.self_blob.area
        # attack weakest adjacent enemy if cheap
        weak = None
        for e in st.enemies:
            if e.area < my_area * 0.5 and len(st.attack_targets):
                weak = e
                break
        if weak is not None and len(st.attack_targets):
            d = (st.attack_targets[:, 0] - weak.centroid[0]) ** 2 + (st.attack_targets[:, 1] - weak.centroid[1]) ** 2
            i = int(np.argmin(d))
            clicks.append(("attack", int(st.attack_targets[i][0]), int(st.attack_targets[i][1]), 10.0))
        elif len(st.expand_targets):
            for i in range(min(self.clicks_per_tick, len(st.expand_targets))):
                t = st.expand_targets[i]
                clicks.append(("expand", int(t[0]), int(t[1]), 0.0))
        return clicks
