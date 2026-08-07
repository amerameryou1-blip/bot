"""Configuration: colors, strategy weights, loop settings.

Everything the game may make you tweak lives here so tuning never means
rewriting code. Edit values, don't edit logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlayerColor:
    """A single player's territory color on screen (RGB)."""
    name: str
    r: int
    g: int
    b: int

    @property
    def rgb(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)


@dataclass
class Palette:
    """Maps color -> who owns it.

    `self_color` is YOUR color. `enemy_colors` is every other player color
    you expect to see. Calibration: take a screenshot, read the colors of the
    territories on screen, and fill them in here.
    """
    self_color: PlayerColor = PlayerColor("me", 230, 80, 80)
    enemy_colors: list[PlayerColor] = field(default_factory=lambda: [
        PlayerColor("blue", 70, 120, 240),
        PlayerColor("green", 70, 200, 120),
        PlayerColor("orange", 240, 160, 60),
        PlayerColor("purple", 170, 90, 220),
        PlayerColor("teal", 60, 190, 190),
    ])
    # Pixels farther than this from every known color are "neutral/unknown".
    tolerance: float = 55.0
    # Work at reduced resolution for speed; segment() output is scaled back up.
    downscale: int = 2

    @property
    def all_colors(self) -> list[PlayerColor]:
        return [self.self_color, *self.enemy_colors]


@dataclass
class StrategyConfig:
    """Decision weights. Higher = the brain cares more about it."""
    # -- expansion --
    expand_weight: float = 1.0
    # Prefer frontier tiles near me (1.0) vs spreading far (0.0).
    frontier_bias: float = 0.6
    # Random nudge applied to expansion so the bot doesn't loop forever on ties.
    wander_noise: float = 0.08
    # -- combat --
    attack_weight: float = 2.4
    # enemy_area / my_area below this -> safe to attack.
    attack_ratio_max: float = 0.75
    # Only attack enemies whose centroid is within this fraction of the screen.
    attack_range: float = 0.30
    # enemy_area / my_area above this -> treat as a threat.
    fear_ratio_min: float = 1.35
    # A threat whose centroid is within this * (threat radius) of my centroid triggers flight.
    threat_radius_factor: float = 1.7
    # Distance my centroid must travel to reach a target (fraction of screen).
    max_expand_dist: float = 0.22


@dataclass
class LoopConfig:
    """Capture -> think -> act cadence."""
    hz: float = 12.0        # ticks per second (10-20 is a good balance)
    log_every: int = 30     # log a line every N ticks
    stats_every: int = 120  # print stats every N ticks
