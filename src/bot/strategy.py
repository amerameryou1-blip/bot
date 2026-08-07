"""The decision brain: FrameState -> heading.

Pure logic, no I/O, so it's fully unit-testable and runs in microseconds.
Priorities: FLEE > ATTACK > EXPAND > WANDER.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import StrategyConfig
from .state import Blob, FrameState


class Priority(Enum):
    WANDER = 0
    EXPAND = 1
    ATTACK = 2
    FLEE = 3


@dataclass
class Decision:
    """One tick of the brain."""
    dx: float                 # -1..1, +x = right
    dy: float                 # -1..1, +y = down
    priority: Priority
    reason: str
    target: tuple[float, float] | None = None  # (row, col) in screen space

    @property
    def angle_deg(self) -> float:
        """Heading angle for logging (0 = up, clockwise)."""
        return float(np.degrees(np.arctan2(self.dx, -self.dy)) % 360)


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


class StrategyBrain:
    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()

    def decide(self, state: FrameState, last_heading: np.ndarray | None = None) -> Decision:
        me = state.self_blob
        h, w = state.shape
        cfg = self.config

        if me is None or me.area == 0:
            return Decision(0.0, 0.0, Priority.WANDER, "no self detected, standing still")

        my_center = np.array([me.centroid[0], me.centroid[1]])
        last = last_heading if last_heading is not None else np.array([0.0, 0.0])

        # ---- FLEE: a much bigger enemy is too close ----
        threats: list[tuple[Blob, float]] = []
        for e in state.enemies:
            if e.area <= 0:
                continue
            ratio = e.area / me.area
            if ratio < cfg.fear_ratio_min:
                continue
            dist = float(np.linalg.norm(np.array([e.centroid[0], e.centroid[1]]) - my_center))
            if dist < cfg.threat_radius_factor * e.radius:
                threats.append((e, ratio))
        if threats:
            # Weighted sum of "run away from each threat" vectors.
            away = np.zeros(2, dtype=float)
            for e, ratio in threats:
                v = my_center - np.array([e.centroid[0], e.centroid[1]])
                away += _norm(v) * ratio
            away = _norm(away)
            return Decision(
                float(away[1]), float(away[0]), Priority.FLEE,
                f"fleeing {len(threats)} bigger enemy(ies)",
                target=(float(my_center[0]), float(my_center[1])),
            )

        # ---- ATTACK: a small enemy is close enough to capture ----
        best_attack: tuple[float, Blob, np.ndarray] | None = None
        for e in state.enemies:
            if e.area <= 0:
                continue
            ratio = e.area / me.area
            if ratio > cfg.attack_ratio_max:
                continue
            v = np.array([e.centroid[0], e.centroid[1]]) - my_center
            dist = float(np.linalg.norm(v))
            max_range = cfg.attack_range * min(h, w)
            if dist > max_range:
                continue
            score = cfg.attack_weight * (1.0 - ratio) / max(dist, 1.0)
            if best_attack is None or score > best_attack[0]:
                best_attack = (score, e, _norm(v))
        if best_attack is not None:
            _, e, dirv = best_attack
            return Decision(
                float(dirv[1]), float(dirv[0]), Priority.ATTACK,
                f"attacking {e.label} (ratio {e.area / max(me.area, 1):.2f})",
                target=(e.centroid[0], e.centroid[1]),
            )

        # ---- EXPAND: move toward the best safe frontier tile ----
        if len(state.frontiers) > 0:
            fs = state.frontiers.astype(float)
            d = fs - my_center[None, :]
            dists = np.linalg.norm(d, axis=1)
            max_d = max(dists.max(), 1e-6)

            # Score: closer frontiers score higher (frontier_bias); also keep
            # away from frontiers that are near a big enemy.
            proximity = 1.0 - (dists / max_d)
            scores = cfg.expand_weight * (cfg.frontier_bias * proximity + (1 - cfg.frontier_bias))

            # Penalize frontiers within threat range of any big enemy.
            for e in state.enemies:
                if e.area <= 0 or e.area < me.area * cfg.fear_ratio_min:
                    continue
                ec = np.array([e.centroid[0], e.centroid[1]])
                near_enemy = np.linalg.norm(fs - ec[None, :], axis=1) < cfg.threat_radius_factor * e.radius
                scores[near_enemy] *= 0.05

            # Small random nudge to break ties / prevent loops.
            scores += cfg.wander_noise * np.random.default_rng().random(len(scores))

            best = int(np.argmax(scores))
            target = fs[best]
            heading = _norm(target - my_center)
            if dists[best] > cfg.max_expand_dist * min(h, w):
                # Far target: still move toward it (heading is enough).
                pass
            return Decision(
                float(heading[1]), float(heading[0]), Priority.EXPAND,
                f"expanding toward frontier #{best} ({dists[best]:.0f}px)",
                target=(float(target[0]), float(target[1])),
            )

        # ---- WANDER: keep last heading so we don't stall ----
        return Decision(
            float(last[1]), float(last[0]), Priority.WANDER,
            "no frontiers found, continuing",
        )
