"""Headless territorial.io simulator — COMBAT-CORRECT.

Real rules modeled (from research):
  - expand: click neutral frontier -> claim at 2 troops/pixel
  - attack: click enemy border with slider % -> spend balance*pct
    defender fights at 2:1 -> capture requires spend > 2*defender_balance
    captured land proportional to excess force; land attack tax ~1.17%
  - elimination: a player at 0 pixels is out; their land becomes NEUTRAL
  - bots full-send sometimes (drain themselves -> vulnerable)
  - win = last survivor
"""
from __future__ import annotations

import numpy as np

from bot.economy import HARD_LIMIT, NEUTRAL_PIXEL_COST, SOFT_LIMIT, TroopTracker
from bot.vision import _adjacent_mask, find_attack_targets, find_expand_targets


class ClickPlayer:
    __slots__ = ("name", "color", "troops", "alive", "rng", "_attack", "last_action")

    def __init__(self, name: str, color: int, seed: int):
        self.name = name
        self.color = color
        self.troops = TroopTracker(balance=512.0, land=12)
        self.alive = True
        self.rng = np.random.default_rng(seed + color)
        self._attack = None  # (target_color, budget, y, x)
        self.last_action = "none"


class ClickSim:
    def __init__(self, h: int = 90, w: int = 125, n_bots: int = 2, seed: int = 1,
                 max_ticks: int = 2000, clicks_per_tick: int = 12, bot_fullsend_chance: float = 0.3):
        self.h, self.w = h, w
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.max_ticks = max_ticks
        self.clicks_per_tick = clicks_per_tick
        self.bot_fullsend_chance = bot_fullsend_chance

        # 0 = neutral land, -1 = water
        self.world = np.zeros((h, w), dtype=np.int16)
        for _ in range(6):
            cy = int(self.rng.integers(0, h))
            cx = int(self.rng.integers(0, w))
            r = int(self.rng.integers(4, 9))
            yy, xx = np.ogrid[:h, :w]
            self.world[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = -1

        self.players: dict[int, ClickPlayer] = {1: ClickPlayer("OURS", 1, seed)}
        for i in range(n_bots):
            self.players[2 + i] = ClickPlayer(f"bot{i}", 2 + i, seed + 100 + i)
        self._pids = sorted(self.players)

        self._place_players()

    def _place_players(self):
        """Spawn on the largest open landmasses, spread apart (the meta:
        open spawn = room to grow)."""
        land = self.world == 0
        # connected components (4-neighbour) of land
        from scipy import ndimage
        lab, n = ndimage.label(land)
        if n == 0:
            return
        sizes = [(int((lab == i).sum()), i) for i in range(1, n + 1)]
        sizes.sort(reverse=True)
        # use the top components; players go on the biggest ones, spread out
        components = [i for _, i in sizes[: max(1, len(self._pids))]]
        used = []
        for pid in self._pids:
            comp = components[(len(used)) % len(components)]
            pts = np.argwhere(lab == comp)
            rng = self.rng
            if used:
                # pick a point far from already-placed players
                best, best_d = None, -1
                for _ in range(60):
                    p = pts[rng.integers(0, len(pts))]
                    d = min(((p[0] - u[0]) ** 2 + (p[1] - u[1]) ** 2) for u in used)
                    if d > best_d:
                        best_d, best = d, p
                y, x = int(best[0]), int(best[1])
            else:
                y, x = int(pts[rng.integers(0, len(pts))][0]), int(pts[rng.integers(0, len(pts))][1])
            used.append((y, x))
            yy, xx = np.ogrid[: self.h, : self.w]
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
        """actions: pid -> list of (kind, y, x, pct)."""
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
                pl.last_action = "expand"
        elif kind == "attack":
            target = int(self.world[y, x])
            if target <= 0 or target == pl.color or target not in self.players:
                return
            budget = pl.troops.attack_budget(pct)
            pl._attack = (target, budget, y, x)

    def _resolve_attacks(self):
        for pid in self._pids:
            pl = self.players[pid]
            if not pl.alive or pl._attack is None:
                continue
            target, budget, y, x = pl._attack
            pl._attack = None
            defender = self.players.get(target)
            if not defender or not defender.alive:
                continue
            # land attack tax ~1.17% of balance
            pl.troops.balance = max(0.0, pl.troops.balance - pl.troops.balance * 0.0117)
            # defender fights at 2:1
            defense = 2.0 * defender.troops.balance
            if budget > defense:
                excess = budget - defense
                # captured land ~ pixels we can afford at 2 troops each
                capture_n = int(excess / (2 * NEUTRAL_PIXEL_COST))
                border = (self.world == target) & _adjacent_mask(self.world == pid)
                idx = np.argwhere(border)
                captured = 0
                if len(idx):
                    n = min(capture_n, len(idx))
                    if n > 0:
                        pick = pl.rng.choice(len(idx), size=n, replace=False)
                        for i in pick:
                            yy, xx = idx[i]
                            self.world[yy, xx] = pid
                            pl.troops.land += 1
                        captured = n
                defender.troops.balance = max(0.0, defender.troops.balance - budget)
                pl.last_action = f"attack_captured_{captured}"
            else:
                # attack failed; defender still loses some from defending
                defender.troops.balance = max(0.0, defender.troops.balance - budget * 0.5)
                pl.last_action = "attack_failed"

    def _eliminate(self):
        for pid in self._pids:
            if (self.world == pid).sum() == 0:
                self.players[pid].alive = False
                # eliminated -> land becomes neutral (free to claim)
                self.world[self.world == pid] = 0

    # -- full match -----------------------------------------------------------

    def run_match(self, our_click_fn, report_every=400):
        """our_click_fn(state) -> ClickAction. Returns result dict.
        Win = LAST SURVIVOR."""
        our_max = 0
        report = []
        while self.tick < self.max_ticks:
            actions: dict[int, list] = {}
            if self.players[1].alive:
                state = self.state_for(1)
                if state.self_blob:
                    # feed enemy balances so the planner can target drained bots
                    if hasattr(our_click_fn, "planner"):
                        balances = {f"e{pid}": self.players[pid].troops.balance
                                    for pid in self._pids if pid != 1 and self.players[pid].alive}
                        our_click_fn.planner.set_enemy_balances(balances)
                    act = our_click_fn(state)
                    actions[1] = self._clicks_for(act)
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
        # strict last-survivor: only a single survivor is a win; stalemates = 0
        winner = alive[0] if len(alive) == 1 else 0
        return {"winner": winner, "our_max_area": our_max,
                "our_final_area": int((self.world == 1).sum()),
                "ticks": self.tick, "alive": alive, "report": report}

    def _clicks_for(self, act):
        """Convert one ClickAction into sim clicks (n per decision)."""
        clicks = []
        if act.kind == "expand":
            st = self.state_for(1)
            targets = st.expand_targets
            if len(targets):
                ty, tx = int(act.y), int(act.x)
                d = (targets[:, 0] - ty) ** 2 + (targets[:, 1] - tx) ** 2
                order = np.argsort(d)[: self.clicks_per_tick]
                for i in order:
                    y, x = int(targets[i][0]), int(targets[i][1])
                    clicks.append(("expand", y, x, 0.0))
        elif act.kind == "attack":
            # ONE attack per decision (the real game rhythm: attack once per
            # income cycle at the slider %).
            st = self.state_for(1)
            if len(st.attack_targets):
                ty, tx = int(act.y), int(act.x)
                d = (st.attack_targets[:, 0] - ty) ** 2 + (st.attack_targets[:, 1] - tx) ** 2
                i = int(np.argmin(d))
                y, x = int(st.attack_targets[i][0]), int(st.attack_targets[i][1])
                clicks.append(("attack", y, x, act.pct))
        return clicks

    def _bot_clicks(self, pid: int) -> list:
        """Bots: expand into neutral while it's plentiful, then attack —
        sometimes FULL-SEND (draining themselves — the meta says attack them
        right after)."""
        pl = self.players[pid]
        st = self.state_for(pid)
        if not st.self_blob:
            return []
        clicks = []
        # expand while there's plenty of neutral in reach; stop when scarce
        if len(st.expand_targets) > 12:
            for i in range(min(self.clicks_per_tick, len(st.expand_targets))):
                t = st.expand_targets[i]
                clicks.append(("expand", int(t[0]), int(t[1]), 0.0))
            return clicks
        # neutral scarce or gone -> attack; sometimes full-send (drain)
        if len(st.attack_targets):
            target = st.attack_targets[0]
            if pl.rng.random() < self.bot_fullsend_chance:
                pct = 90.0  # full send -> drains itself
            else:
                pct = 15.0
            clicks.append(("attack", int(target[0]), int(target[1]), pct))
            return clicks
        # nothing to do but expand
        if len(st.expand_targets):
            t = st.expand_targets[0]
            clicks.append(("expand", int(t[0]), int(t[1]), 0.0))
        return clicks
