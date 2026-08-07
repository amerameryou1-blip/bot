"""Click-based simulator v5 — attack-meta bots + screen rendering for the CNN.

Upgrades over game4:
  - Bots follow the meta: expand to free land, then attack the weakest adjacent
    enemy with a %; occasionally they "full send" (big %), which leaves them
    weak right after — the exact exploit the neural net should learn.
  - Tracks enemy balance + when they last attacked (balance delta).
  - `render_rgb(pid)` produces a screen-like RGB image + per-pixel labels, so
    the vision CNN can learn image -> map from the simulator.
  - Win = last survivor (true territorial.io win condition).
"""
from __future__ import annotations

import numpy as np

from bot.economy import NEUTRAL_PIXEL_COST, TroopTracker
from bot.vision import _adjacent_mask, find_attack_targets, find_expand_targets


class ClickPlayer:
    __slots__ = ("name", "color", "troops", "alive", "rng", "_pending_attack",
                 "last_attack_tick", "bank_ticks", "strategy")

    def __init__(self, name: str, color: int, seed: int, strategy: str = "medium"):
        self.name = name
        self.color = color
        self.troops = TroopTracker(balance=512.0, land=12)
        self.alive = True
        self.rng = np.random.default_rng(seed + color)
        self._pending_attack = None
        self.last_attack_tick = -999
        self.bank_ticks = 0
        self.strategy = strategy  # easy / medium / hard


class ClickSim5:
    def __init__(self, h: int = 130, w: int = 170, n_bots: int = 3, seed: int = 1,
                 max_ticks: int = 1800, clicks_per_tick: int = 12, bot_skill: str = "medium"):
        self.h, self.w = h, w
        self.bot_skill = bot_skill
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
            self.players[2 + i] = ClickPlayer(f"bot{i}", 2 + i, seed + 100 + i, strategy=bot_skill)
        self._pids = sorted(self.players)

        spots = np.argwhere(self.world == 0)
        self.rng.shuffle(spots)
        for i, pid in enumerate(self._pids):
            if i >= len(spots):
                break
            y, x = int(spots[i][0]), int(spots[i][1])
            yy, xx = np.ogrid[:h, :w]
            self.world[(yy - y) ** 2 + (xx - x) ** 2 < 4 ** 2] = pid

        # enemy colors for rendering (vivid, distinct)
        self.enemy_colors = {pid: (60, 140, 240) if pid % 3 == 0 else
                             (60, 200, 120) if pid % 3 == 1 else (240, 160, 60)
                             for pid in self._pids}

    # -- observation -------------------------------------------------------

    def state_for(self, pid: int = 1):
        from bot.state import Blob, FrameState
        # labels: -1 water, 0 neutral land, 1 me, 2+ enemies (water != expandable)
        labels = np.where(self.world < 0, -1,
                          np.where(self.world == pid, 1, np.where(self.world > 0, 2, 0))).astype(np.int8)
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

    # -- rendering (for the vision CNN) --------------------------------------

    def render_rgb(self, pid: int = 1, scale: int = 4, noise: float = 6.0):
        """Render a screen-like RGB image (H/scale x W/scale x 3) + labels.

        labels: 0 water, 1 neutral, 2 me, 3 enemy.
        Colors mimic the real game (blue water, beige land, vivid territories).
        """
        h, w = self.h // scale, self.w // scale
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        labels = np.zeros((h, w), dtype=np.int64)
        for y in range(h):
            for x in range(w):
                v = self.world[y * scale, x * scale]
                if v < 0:
                    rgb[y, x] = (40, 90, 170)
                    labels[y, x] = 0
                elif v == 0:
                    rgb[y, x] = (150, 142, 120)
                    labels[y, x] = 1
                elif v == pid:
                    rgb[y, x] = (220, 60, 60)
                    labels[y, x] = 2
                else:
                    rgb[y, x] = self.enemy_colors.get(v, (200, 60, 200))
                    labels[y, x] = 3
        if noise > 0:
            rgb = np.clip(rgb.astype(int) + self.rng.normal(0, noise, rgb.shape), 0, 255).astype(np.uint8)
        return rgb, labels

    def frame_tensor(self, pid: int = 1, size: int = 64):
        """(rgb[64,64,3], labels[64,64]) for the CNN — fast path."""
        rgb, labels = self.render_rgb(pid)
        if rgb.shape[0] != size:
            from PIL import Image
            img = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
            rgb = np.array(img)
            from PIL import Image as I2
            lbl = I2.fromarray(labels.astype(np.uint8)).resize((size, size), I2.NEAREST)
            labels = np.array(lbl)
        return rgb, labels

    # -- stepping ------------------------------------------------------------

    def step(self, actions: dict[int, list[tuple[str, int, int, float]]]):
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
            budget = pl.troops.attack_budget(pct)
            # SENDING troops costs them (they're gone) — the economy matters.
            pl.troops.balance = max(0.0, pl.troops.balance - budget)
            pl._pending_attack = (target, budget, y, x)
            pl.last_attack_tick = self.tick

    def _resolve_attacks(self):
        for pid in self._pids:
            pl = self.players[pid]
            if not pl.alive:
                continue
            pending = getattr(pl, "_pending_attack", None)
            if not pending:
                continue
            target, budget, y, x = pending
            del pl._pending_attack
            defender = self.players.get(target)
            if not defender or not defender.alive:
                continue
            dbal = defender.troops.balance
            # 2:1 combat: my sent force fights at half effectiveness vs their
            # balance. Win (conquer) iff budget >= 2 * dbal.
            if budget >= 2.0 * dbal:
                # defender wiped: capture their border AND all territory
                defender.troops.balance = 0.0
                self.world[self.world == target] = pid
                pl.troops.land = int((self.world == pid).sum())
            else:
                # failed attack: defender repels, but loses troops doing so
                defender.troops.balance = max(0.0, dbal - budget / 2.0)
                # border skirmish: a small slice flips if we got close
                if budget >= 1.5 * dbal:
                    border = (self.world == target) & _adjacent_mask(self.world == pid)
                    idx = np.argwhere(border)
                    if len(idx):
                        n = min(3, len(idx))
                        pick = pl.rng.choice(len(idx), size=n, replace=False)
                        for i in pick:
                            yy, xx = idx[i]
                            self.world[yy, xx] = pid
                            pl.troops.land += 1

    def _eliminate(self):
        for pid in self._pids:
            if (self.world == pid).sum() == 0:
                self.players[pid].alive = False

    # -- full match -----------------------------------------------------------

    def run_match(self, our_click_fn, report_every=400):
        """Win = LAST SURVIVOR. Returns result dict incl. our rank."""
        our_max = 0
        report = []
        while self.tick < self.max_ticks:
            actions: dict[int, list] = {}
            if self.players[1].alive:
                state = self.state_for(1)
                if state.self_blob:
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
        # our rank by area among alive+dead
        areas = {pid: int((self.world == pid).sum()) for pid in self._pids}
        rank = 1 + sum(1 for pid, a in areas.items() if pid != 1 and a > areas[1])
        return {"winner": winner, "our_rank": rank, "our_max_area": our_max,
                "our_final_area": int((self.world == 1).sum()),
                "ticks": self.tick, "alive": alive, "report": report}

    def _clicks_for(self, act, n):
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

    # -- meta bots -------------------------------------------------------------

    def _bot_clicks(self, pid: int) -> list:
        pl = self.players[pid]
        st = self.state_for(pid)
        if not st.self_blob:
            return []
        my_area = st.self_blob.area
        density = pl.troops.balance / max(my_area, 1)
        skill = pl.strategy
        cpt = self.clicks_per_tick
        if skill == "easy":
            cpt = max(3, int(cpt * 0.5))
        elif skill == "hard":
            cpt = min(24, int(cpt * 1.4))
        clicks = []

        # weakest adjacent enemy by balance
        weak_now = None
        if len(st.attack_targets):
            best_bal = float("inf")
            for e in st.enemies:
                bal = self.players[int(e.label[1:])].troops.balance
                if bal < best_bal:
                    best_bal = bal
                    weak_now = (e, bal)

        # opportunistic attack on a very weak neighbor (all skills, tuned per skill)
        if weak_now and weak_now[1] < pl.troops.balance * 0.15:
            chance = 0.35 if skill == "easy" else (0.7 if skill == "medium" else 0.9)
            if pl.rng.random() < chance:
                e, _ = weak_now
                d = (st.attack_targets[:, 0] - e.centroid[0]) ** 2 + (st.attack_targets[:, 1] - e.centroid[1]) ** 2
                i = int(np.argmin(d))
                pct = 6.0 + pl.rng.random() * 4.0
                clicks.append(("attack", int(st.attack_targets[i][0]), int(st.attack_targets[i][1]), pct))
                return clicks

        # expand free land
        if len(st.expand_targets) > 0:
            for i in range(min(cpt, len(st.expand_targets))):
                t = st.expand_targets[i]
                clicks.append(("expand", int(t[0]), int(t[1]), 0.0))
            return clicks

        # combat phase: attack weakest
        if weak_now is not None:
            e, _ = weak_now
            d = (st.attack_targets[:, 0] - e.centroid[0]) ** 2 + (st.attack_targets[:, 1] - e.centroid[1]) ** 2
            i = int(np.argmin(d))
            if skill == "easy":
                if pl.rng.random() < 0.5:
                    return []  # easy bots sometimes waste a turn
                pct = 6.0 + pl.rng.random() * 4.0
            elif skill == "hard":
                # hard bots attack more decisively and more often
                if density > 70 or weak_now[1] < pl.troops.balance * 0.2:
                    pct = 12.0 + pl.rng.random() * 6.0
                else:
                    return []
            else:
                if density > 120 and pl.rng.random() < 0.5:
                    pct = 45.0  # medium full-send -> weak window
                else:
                    pct = 8.0 + pl.rng.random() * 6.0
            clicks.append(("attack", int(st.attack_targets[i][0]), int(st.attack_targets[i][1]), pct))
        return clicks
