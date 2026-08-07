"""Unit tests for the strategy brain — pure logic, no browser needed."""
from __future__ import annotations

import numpy as np

import pytest

from bot.config import StrategyConfig
from bot.strategy import Priority, StrategyBrain
from sim.arena import build_state


@pytest.fixture
def brain() -> StrategyBrain:
    return StrategyBrain()


def test_expands_when_alone(brain: StrategyBrain) -> None:
    state = build_state()
    decision = brain.decide(state)
    assert decision.priority is Priority.EXPAND
    # Heading should point from my centroid toward some frontier (nonzero).
    assert (decision.dx, decision.dy) != (0.0, 0.0)
    assert "expand" in decision.reason.lower()


def test_flees_from_big_enemy(brain: StrategyBrain) -> None:
    # Big enemy right on top of me.
    state = build_state(
        enemies=[((125.0, 165.0), 45.0)],  # area ~ 6300 vs me ~ 2800 => ratio > fear_ratio_min
    )
    decision = brain.decide(state)
    assert decision.priority is Priority.FLEE
    # Heading must point AWAY from the enemy: enemy at (125,165), me at (120,160).
    # Away vector points up-left => dx negative, dy negative.
    assert decision.dx < 0
    assert decision.dy < 0


def test_attacks_small_enemy_when_close(brain: StrategyBrain) -> None:
    state = build_state(
        enemies=[((125.0, 175.0), 12.0)],  # small, close
    )
    decision = brain.decide(state)
    assert decision.priority is Priority.ATTACK
    # Toward the enemy (to the right / slightly down).
    assert decision.dx > 0


def test_ignores_far_small_enemy_when_expansion_is_better(brain: StrategyBrain) -> None:
    # Small enemy very far away -> out of attack range -> expand instead.
    state = build_state(
        enemies=[((20.0, 30.0), 10.0)],
    )
    decision = brain.decide(state)
    assert decision.priority is not Priority.ATTACK
    assert decision.priority is Priority.EXPAND


def test_wanders_when_no_frontiers(brain: StrategyBrain) -> None:
    # Build a state where I own the entire map -> no frontier.
    from sim.arena import build_state
    from bot.state import FrameState, Blob

    h, w = 80, 80
    mask = np.ones((h, w), dtype=bool)
    me = Blob("me", mask, h * w, (h / 2, w / 2))
    state = FrameState(shape=(h, w), labels=np.ones((h, w), dtype=int), self_blob=me, enemies=[], frontiers=np.zeros((0, 2), dtype=int))
    decision = brain.decide(state)
    assert decision.priority is Priority.WANDER


def test_expansion_avoids_frontier_near_big_enemy(brain: StrategyBrain) -> None:
    """When two frontiers exist, one near a big enemy, prefer the safe one."""
    h, w = 300, 300
    from bot.state import Blob, FrameState

    # Me: vertical stripe down the middle-left.
    labels = np.zeros((h, w), dtype=int)
    labels[20:280, 130:150] = 1

    # Big enemy touching the RIGHT edge of my stripe (around x=150..).
    labels[40:120, 150:210] = 2

    # Frontier points: some on the left (safe), some adjacent to the enemy (right side).
    me_mask = labels == 1
    neutral = labels == 0
    from bot.vision import _find_frontiers

    frontiers = _find_frontiers(me_mask, neutral, max_samples=1000)

    # Must contain points both to the left of me and hugging the enemy on the right.
    left_f = frontiers[frontiers[:, 1] <= 130]  # my left edge (col 130)
    near_f = frontiers[frontiers[:, 1] >= 149]  # my right edge (col 149) touches the enemy
    assert len(left_f) > 0 and len(near_f) > 0

    me_coords = np.argwhere(me_mask)
    me = Blob("me", me_mask, len(me_coords), (float(me_coords[:, 0].mean()), float(me_coords[:, 1].mean())))
    enemy_mask = labels == 2
    ec = np.argwhere(enemy_mask)
    enemy = Blob("enemy2", enemy_mask, len(ec), (float(ec[:, 0].mean()), float(ec[:, 1].mean())))

    state = FrameState(shape=(h, w), labels=labels, self_blob=me, enemies=[enemy], frontiers=frontiers)
    decision = brain.decide(state)

    # Should NOT attack (enemy is much bigger than me here), and should expand
    # somewhere — the scoring should steer clear of the enemy side.
    assert decision.priority in (Priority.EXPAND, Priority.FLEE)
    if decision.priority is Priority.EXPAND:
        # Expanding: chosen target should be on the left (away from the enemy).
        assert decision.target is not None
        assert decision.target[1] < 150


def test_flee_beats_attack_when_enemy_is_both_small_and_big_ratio(brain: StrategyBrain) -> None:
    """Regression: an enemy just under the attack ratio but huge in absolute
    area should still be attacked only when ratio says safe. Here we make one
    clearly bigger so FLEE wins over ATTACK."""
    state = build_state(
        enemies=[((124.0, 164.0), 40.0)],
    )
    decision = brain.decide(state)
    assert decision.priority is Priority.FLEE
