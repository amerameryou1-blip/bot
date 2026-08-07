"""Troop economy — exact numbers from the player's mechanics doc.

- Interest tick every 0.56 s (1 tick): balance *= (1 + rate/100)
- Income tick every 5.6 s (10 ticks): balance += land pixels
- Base interest 1–7% early; boosted up to 7× for the first 107 s
- Soft limit 100 troops/pixel → interest decays above it
- Hard limit 150 troops/pixel → interest 0, balance capped
- Start: 512 troops, 12 land pixels
"""
from __future__ import annotations

TICK_S = 0.56
BOOST_S = 107.0
BOOST_TICKS = int(BOOST_S / TICK_S)  # ~191 ticks of initial augmentation
SOFT_LIMIT = 100.0
HARD_LIMIT = 150.0
NEUTRAL_PIXEL_COST = 2.0  # minimal troop cost to claim a neutral pixel


class TroopTracker:
    """Models the balance so the planner knows when to expand / bank / attack.

    When the leaderboard is OCR'd (it shows balance + land), the caller can
    overwrite `balance` with the true value; otherwise the model estimates it.
    """

    def __init__(self, balance: float = 512.0, land: int = 12, tick: int = 0,
                 initial_troops: float | None = None):
        # backward-compat: old callers used initial_troops=
        if initial_troops is not None:
            balance = initial_troops
        self.balance = float(balance)
        self.land = max(land, 1)
        self.tick = tick

    def density(self) -> float:
        return self.balance / self.land

    def base_rate(self) -> float:
        """Decays 7% → 1% over ~20 min (TerriEngine formula)."""
        pct = 100 * (13440 - 6 * self.tick) // 1920
        return max(1.0, min(7.0, pct)) / 100.0

    def rate(self) -> float:
        r = self.base_rate()
        d = self.density()
        if d >= HARD_LIMIT:
            return 0.0
        if d > SOFT_LIMIT:
            # interest decays as density exceeds the soft limit
            r *= (HARD_LIMIT - d) / (HARD_LIMIT - SOFT_LIMIT)
        if self.tick < BOOST_TICKS:
            r = min(r * 7.0, 0.60)  # up to 7× early; sanity cap 60%/tick
        return max(0.0, r)

    def update(self, land: int) -> None:
        """Advance one 0.56 s tick with the observed land area."""
        self.land = max(int(land), 1)
        self.tick += 1
        self.balance *= 1.0 + self.rate()
        if self.tick % 10 == 0:  # every 5.6 s
            self.balance += self.land
        cap = HARD_LIMIT * self.land
        if self.balance > cap:
            self.balance = cap

    # -- backward-compat with the old API -----------------------------------

    @property
    def troops(self) -> float:
        """Old name for balance."""
        return self.balance

    @troops.setter
    def troops(self, value: float) -> None:
        self.balance = float(max(0.0, value))

    def can_claim(self, pixels: int) -> bool:
        """Old API: can we afford to claim `pixels` neutral pixels?"""
        return self.balance >= NEUTRAL_PIXEL_COST * pixels

    def claim(self, pixels: int) -> int:
        """Old API: claim up to `pixels` at NEUTRAL_PIXEL_COST each."""
        affordable = int(self.balance // NEUTRAL_PIXEL_COST)
        claimed = min(int(pixels), affordable)
        self.balance -= NEUTRAL_PIXEL_COST * claimed
        self.land += claimed
        return claimed

    # -- spending ----------------------------------------------------------

    def attack_budget(self, pct: float) -> float:
        return self.balance * max(0.0, min(100.0, pct)) / 100.0

    def can_afford_expansion(self, pixels: int = 1) -> bool:
        return self.balance >= NEUTRAL_PIXEL_COST * pixels

    def spend_expansion(self, pixels: int) -> None:
        cost = NEUTRAL_PIXEL_COST * pixels
        self.balance = max(0.0, self.balance - cost)
        self.land += pixels

    def snapshot(self) -> dict:
        return {
            "balance": round(self.balance, 1),
            "land": self.land,
            "density": round(self.density(), 2),
            "rate_pct": round(self.rate() * 100, 3),
            "tick": self.tick,
        }
