"""Lookahead planner — "thinks" like a chess bot over candidate headings.

For every frame, the planner:
  1. Builds an occupancy map (neutral / mine / enemy).
  2. Generates N candidate headings (16 compass directions).
  3. For each, projects the expansion: how much neutral land the current troop
     budget can claim along that beam, and how dangerous that beam is
     (enemy proximity + size).
  4. Re-evaluates the top candidates one ply deeper (expand-then-reconsider)
     so the chosen move leads to a good *next* position, not just a good now.
  5. Applies hysteresis to avoid dithering between similar headings.

Pure numpy + the economy model; no I/O — fully unit-testable.
"""
from __future__ import annotations

import numpy as np

from .economy import TroopTracker
from .state import Blob, FrameState

SEARCH_DIRS = 16  # compass directions sampled per frame
LOOKAHEAD_PLY = 1  # how many extra re-evaluation steps (depth beyond 0)
HYSTERESIS = 0.85  # keep previous heading if within this fraction of best score


class PlannerConfig:
    def __init__(
        self,
        gain_weight: float = 1.0,
        risk_weight: float = 2.2,
        attack_weight: float = 3.0,
        corner_penalty: float = 0.35,
        max_beam_len: float = 0.34,      # as fraction of min(h, w)
        beams: int = SEARCH_DIRS,
        attack_ratio_max: float = 0.6,   # only attack enemies smaller than this × me
    ) -> None:
        self.gain_weight = gain_weight
        self.risk_weight = risk_weight
        self.attack_weight = attack_weight
        self.corner_penalty = corner_penalty
        self.max_beam_len = max_beam_len
        self.beams = beams
        self.attack_ratio_max = attack_ratio_max


class PlanResult:
    """Decision object — matches the interface BotLoop expects (dx, dy, reason)."""

    __slots__ = ("dx", "dy", "score", "label", "reason", "beams")

    def __init__(self, dx: float, dy: float, score: float, label: str, beams: list) -> None:
        self.dx = dx
        self.dy = dy
        self.score = score
        self.label = label
        self.reason = label  # BotLoop logs/records `reason`
        self.beams = beams


# ============================================================================
# Click-based brain — plays the game the way it actually works
# (click-to-expand + attack slider). This is what the notebook uses.
# ============================================================================


class ClickAction:
    """One decision: WHAT to click and HOW."""

    __slots__ = ("kind", "x", "y", "pct", "reason")

    def __init__(self, kind: str, x: float = 0.0, y: float = 0.0, pct: float = 0.0, reason: str = ""):
        self.kind = kind          # 'expand' | 'attack' | 'bank' | 'boat' | 'peace'
        self.x = x                # screen x to click
        self.y = y                # screen y to click
        self.pct = pct            # attack slider % (attacks/boats)
        self.reason = reason

    @property
    def reason_label(self) -> str:
        return f"{self.kind}:{self.reason}"


class ClickPlannerConfig:
    def __init__(
        self,
        expand_pct: float = 12.0,          # slider % when expanding into neutral
        attack_pct: float = 12.0,          # slider % for attacks (meta: 5-20)
        weak_balance_ratio: float = 0.35,  # enemy balance < this × mine = drained/weak
        attack_balance_ratio: float = 2.0, # my balance > this × enemy = have 2:1 advantage
        attack_density: float = 75.0,      # density at which we start attacking (near red ~100)
        spend_density: float = 90.0,       # expand-to-spend threshold
        capped_density: float = 130.0,     # must-spend threshold
        expand_radius: int = 14,           # neutral-richness radius for expand target
    ) -> None:
        self.expand_pct = expand_pct
        self.attack_pct = attack_pct
        self.weak_balance_ratio = weak_balance_ratio
        self.attack_balance_ratio = attack_balance_ratio
        self.attack_density = attack_density
        self.spend_density = spend_density
        self.capped_density = capped_density
        self.expand_radius = expand_radius


class ClickPlanner:
    """Combat brain — expand cheap, then ATTACK to eliminate, bank to red.

    Priority (last-survivor meta):
      1. near hard cap       -> MUST spend: attack best target, else expand
      2. strong + target     -> attack the weakest/drained neighbor at attack_pct
      3. neutral available   -> expand (cheap land first, per the meta)
      4. drained target      -> exploit it even before red interest
      5. else                -> bank (compound toward red interest ~100/px)

    Enemy balances come from `set_enemy_balances` (sim feeds exact values;
    live play OCRs the leaderboard). Without them we fall back to area ratio.
    """

    def __init__(self, config: ClickPlannerConfig | None = None, tracker: TroopTracker | None = None):
        self.config = config or ClickPlannerConfig()
        self.tracker = tracker or TroopTracker()
        self.enemy_balances: dict[str, float] = {}

    def set_enemy_balances(self, balances: dict[str, float]) -> None:
        self.enemy_balances = dict(balances)

    def decide(self, state: FrameState) -> ClickAction:
        me = state.self_blob
        if me is None or me.area == 0:
            return ClickAction("bank", reason="no-self")
        self.tracker.update(me.area)

        cfg = self.config
        balance = self.tracker.balance
        land = max(me.area, 1)
        density = balance / land
        target = self._pick_target(state)

        # 1) near hard cap — MUST spend troops (interest dead anyway)
        if density >= cfg.capped_density:
            if target is not None:
                return ClickAction("attack", target[1], target[2], cfg.attack_pct,
                                   reason=f"capped->attack({target[2]})")
            if len(state.expand_targets):
                t = self._pick_expand_target(state)
                return ClickAction("expand", t[1], t[0], reason=f"capped->expand(d={density:.0f})")
            return ClickAction("bank", reason=f"capped(d={density:.0f})")

        # 2) a DRAINED enemy (e.g. bot just full-sent) — exploit even early.
        #    This is THE meta kill: attack right after they overspend.
        if target is not None and self._is_drained(target):
            return ClickAction("attack", target[1], target[2], cfg.attack_pct,
                               reason=f"drained({target[2]})")

        # 3) we have an attackable target and are at/near the attack density
        if target is not None and density >= cfg.attack_density:
            return ClickAction("attack", target[1], target[2], cfg.attack_pct,
                               reason=f"attack({target[2]},d={density:.0f})")

        # 4) cheap neutral land first (the meta: expand before attacking)
        if len(state.expand_targets) and density < cfg.spend_density:
            t = self._pick_expand_target(state)
            return ClickAction("expand", t[1], t[0], reason=f"grow(d={density:.0f})")

        # 5) above spend density — convert balance to land
        if density >= cfg.spend_density:
            if len(state.expand_targets):
                t = self._pick_expand_target(state)
                return ClickAction("expand", t[1], t[0], reason=f"spend(d={density:.0f})")
            if target is not None:
                return ClickAction("attack", target[1], target[2], cfg.attack_pct,
                                   reason=f"spend->attack({target[2]})")

        # 6) bank and compound toward red interest
        return ClickAction("bank", reason=f"idle(d={density:.0f})")

    # -- target selection ---------------------------------------------------

    def _pick_target(self, state: FrameState):
        """Best adjacent enemy to attack, or None.

        We need a shared border (attack targets exist) AND the balance
        advantage (~2:1 per the defender rule). Among valid targets, pick the
        one with the LOWEST balance (the drained one — easiest kill)."""
        if not state.attack_targets.any():
            return None
        me_balance = self.tracker.balance
        cfg = self.config
        best = None  # (blob, target_y, target_x, label)
        best_bal = float("inf")
        for e in state.enemies:
            if e.area <= 0:
                continue
            bal = self.enemy_balances.get(e.label) if self.enemy_balances else None
            if bal is not None:
                # need ~2:1 advantage (defender bonus) unless enemy is drained
                if bal > me_balance * cfg.attack_balance_ratio and bal > me_balance * cfg.weak_balance_ratio:
                    continue
                score = bal
            else:
                # fallback: small area ratio
                if e.area > state.self_blob.area * 0.6:
                    continue
                score = e.area
            if score < best_bal:
                best_bal = score
                ty, tx = self._border_point(state, e)
                if ty is None:
                    continue
                best = (e, tx, ty, e.label)
        return best

    def _is_drained(self, target) -> bool:
        if not self.enemy_balances:
            return False
        bal = self.enemy_balances.get(target[3])
        return bal is not None and bal < self.tracker.balance * self.config.weak_balance_ratio

    def _border_point(self, state: FrameState, enemy: Blob):
        """A pixel of `enemy` adjacent to me, closest to my centroid."""
        me = state.self_blob
        cy, cx = me.centroid
        targets = state.attack_targets
        best_idx, best_d = None, float("inf")
        for i, (ty, tx) in enumerate(targets):
            # must be on this enemy's border (approx: near its centroid)
            if abs(ty - enemy.centroid[0]) > 80 or abs(tx - enemy.centroid[1]) > 80:
                continue
            d = (ty - cy) ** 2 + (tx - cx) ** 2
            if d < best_d:
                best_d, best_idx = d, i
        if best_idx is None:
            return None, None
        ty, tx = targets[best_idx]
        return float(ty), float(tx)

    def _pick_expand_target(self, state: FrameState) -> tuple[float, float]:
        """Expand target with the richest neutral neighborhood."""
        targets = state.expand_targets
        h, w = state.shape
        neutral = state.neutral_mask
        radius = int(self.config.expand_radius)
        best_idx, best_score = 0, -1
        for i, (ty, tx) in enumerate(targets):
            y0, y1 = max(0, int(ty) - radius), min(h, int(ty) + radius)
            x0, x1 = max(0, int(tx) - radius), min(w, int(tx) + radius)
            score = int(neutral[y0:y1, x0:x1].sum())
            if score > best_score:
                best_score, best_idx = score, i
        ty, tx = targets[best_idx]
        return float(ty), float(tx)

    def set_observed_balance(self, balance: float) -> None:
        """Overwrite the estimated balance with the leaderboard OCR value."""
        self.tracker.balance = float(balance)


class TerritoryPlanner:
    def __init__(self, config: PlannerConfig | None = None, tracker: TroopTracker | None = None) -> None:
        self.config = config or PlannerConfig()
        self.tracker = tracker or TroopTracker()
        self.last_heading: tuple[float, float] = (1.0, 0.0)  # default: right

    def decide(self, state: FrameState, last_heading=None) -> PlanResult:
        me = state.self_blob
        h, w = state.shape
        cfg = self.config

        if me is None or me.area == 0:
            return PlanResult(0.0, 0.0, 0.0, "no-self", [])

        self.tracker.update(me.area)
        center = np.array([me.centroid[0], me.centroid[1]])
        my_area = max(me.area, 1)

        occ = state.labels.copy()
        enemy_mask = occ >= 2
        neutral_mask = occ == 0

        max_len = cfg.max_beam_len * min(h, w)
        budget = self.tracker.troops / 2.0  # pixels claimable right now

        angles = np.linspace(0, 2 * np.pi, cfg.beams, endpoint=False)
        beams: list[dict] = []

        for a in angles:
            dx, dy = float(np.cos(a)), float(np.sin(a))
            direc = np.array([dx, dy])
            open_len, hit_enemy = self._probe_beam(center, direc, max_len, occ, neutral_mask, enemy_mask)
            claimed = min(open_len, budget)

            # Which enemy (if any) is at the beam's end?
            hit_blob = self._enemy_at(center, direc, open_len, state.enemies) if hit_enemy else None
            ratio = hit_blob.area / my_area if hit_blob else 0.0

            risk = self._beam_risk(center, direc, open_len, enemy_mask, h, w)
            target = center + direc * min(open_len, max_len)
            corner = self._corner_penalty(target, h, w)

            attackable = hit_blob is not None and ratio < cfg.attack_ratio_max and hit_blob.area > 50
            if attackable:
                # Attack corridor: reward the approach + the kill, scaled by
                # how much smaller the enemy is.
                score = cfg.attack_weight * min(open_len, budget) * (1.0 - ratio) - cfg.risk_weight * risk
            elif hit_blob is not None:
                # Big enemy wall: expansion up to their border is still useful,
                # but heavily discounted + extra risk (they may attack back).
                score = (cfg.gain_weight * claimed - cfg.risk_weight * risk * 1.6
                         - cfg.corner_penalty * corner - min(ratio, 10.0) * 3.0)
            else:
                score = cfg.gain_weight * claimed - cfg.risk_weight * risk - cfg.corner_penalty * corner

            beams.append({"angle": a, "dx": dx, "dy": dy, "open": open_len, "claimed": claimed,
                          "risk": risk, "score": score, "hit_enemy": hit_enemy, "ratio": ratio})

        # ---- one-ply re-evaluation of the top beams ----
        beams.sort(key=lambda b: b["score"], reverse=True)
        top = beams[: max(1, cfg.beams // 4)]
        for b in top:
            future = self._future_gain(b, center, occ, neutral_mask, enemy_mask, budget)
            b["score"] = 0.7 * b["score"] + 0.3 * future
        beams.sort(key=lambda b: b["score"], reverse=True)

        best = beams[0]

        # Hysteresis: avoid dithering between near-equal headings.
        if self.last_heading != (0.0, 0.0):
            lx, ly = self.last_heading
            for b in beams:
                if abs(b["dx"] - lx) < 0.3 and abs(b["dy"] - ly) < 0.3:
                    if b["score"] >= best["score"] * HYSTERESIS:
                        best = b
                    break

        self.last_heading = (best["dx"], best["dy"])
        label = "attack-corridor" if (best["hit_enemy"] and best["ratio"] < self.config.attack_ratio_max) else "expand"
        return PlanResult(best["dx"], best["dy"], best["score"], label, beams)

    # -- internals -----------------------------------------------------------

    def _probe_beam(self, center, direc, max_len, occ, neutral, enemy):
        """Walk along the beam from my centroid; skip own territory, then
        measure the open neutral run (and whether it ends at an enemy)."""
        h, w = occ.shape
        open_len = 0.0
        hit_enemy = False
        step = 0.0
        while step < max_len:
            pos = center + direc * (step + 1.0)
            r, c = int(round(pos[0])), int(round(pos[1]))
            if not (0 <= r < h and 0 <= c < w):
                break
            v = occ[r, c]
            if v == 1:  # my own territory — skip, we're looking beyond the border
                step += 1.0
                continue
            if enemy[r, c]:
                hit_enemy = True
                break
            if not neutral[r, c]:
                break
            open_len = step + 1.0
            step += 1.0
        return open_len, hit_enemy

    def _enemy_at(self, center, direc, open_len, enemies: list[Blob]) -> Blob | None:
        """Which enemy blob is at the beam's tip (within its radius)?"""
        tip = center + direc * max(open_len, 1.0)
        best: Blob | None = None
        best_dist = float("inf")
        for e in enemies:
            ec = np.array([e.centroid[0], e.centroid[1]])
            dist = float(np.linalg.norm(tip - ec))
            if dist < e.radius and dist < best_dist:
                best_dist = dist
                best = e
        return best

    def _beam_risk(self, center, direc, open_len, enemy_mask, h, w):
        """Enemy mass within a radius around the beam's tip / path."""
        if open_len <= 1:
            return 1.0
        tip = center + direc * open_len
        r0, r1 = max(0, int(tip[0]) - 40), min(h, int(tip[0]) + 40)
        c0, c1 = max(0, int(tip[1]) - 40), min(w, int(tip[1]) + 40)
        if r1 <= r0 or c1 <= c0:
            return 0.0
        mass = int(enemy_mask[r0:r1, c0:c1].sum())
        return mass / max((r1 - r0) * (c1 - c0), 1) * 60.0

    def _corner_penalty(self, target, h, w):
        dx_edge = min(target[1], w - target[1]) / w
        dy_edge = min(target[0], h - target[0]) / h
        return max(0.0, 1.0 - 3.0 * min(dx_edge, dy_edge))

    def _future_gain(self, beam, center, occ, neutral, enemy, budget):
        """After claiming the beam's corridor, how much more land is reachable?"""
        claim = max(1, int(beam["claimed"]))
        tip = center + np.array([beam["dy"], beam["dx"]]) * beam["open"]
        open_ahead, _ = self._probe_beam(tip, np.array([beam["dx"], beam["dy"]]), 80.0, occ, neutral, enemy)
        return min(open_ahead + claim * 0.5, budget + claim)
