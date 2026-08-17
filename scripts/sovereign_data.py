"""
data_adapters.py — shard → SOVEREIGN v3 training samples.

Reads amer224/territorial-bot-data `rl/shards_v2/*.npz`:
    rgb   (N, 256, 256, 3) uint8        frames (recorded every 2 ticks)
    lab   (N, 256, 256) int             {0 water, 1 neutral, 2 me, 3 enemy}
    nums  (N, 8) float                  context vector
    kind  (N,) {0 expand, 1 attack, 2 bank}
    cell  (N,) int                       32x32 grid idx of the click
    pct   (N,) float in [0,1]           troops committed
    logp  (N,) float                    behavior logprob (unused)
    reward(N,) float                    farmer shaping (unknown scale)
    alive (N,) bool                     still alive at this frame
    lens  (list of int)                 per-episode frame counts

v3 extras (all derived from FREE labels already in the shards):

  1. RETURN CONDITIONING (RPO): rtg = z-scored remaining return; rtg_next
     = the shifted target (econ forecaster predicts ctx_full = nums+rtg).
  2. OPPONENT-INTENT MIRRORS: lab[t] vs lab[t+1] deltas give
        threat = pixels that BECOME enemy next frame   (where they strike)
        expand = pixels that BECOME mine next frame    (where I grow)
     These are the training targets for the model's threat/expand heads —
     implicit opponent modeling from consecutive frames.
  3. NEXT-SEGMENTATION: lab64_next = downsampled lab[t+1] (one-step spatial
     dynamics supervision; the seed of a future MCTS world model).
  4. WIN LABEL: 1 if the episode survived to timeout, 0 if it died.
  5. SEQUENCE BUILDER for the GRU memory core (truncated-BPTT windows).

No exotic dependencies (numpy + torch).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader

GAMMA = 0.995
GRID = 32
SURVIVOR_MULT = float(os.environ.get("SOV_SURVIVOR_MULT", "4.0"))
KILL_MULT = float(os.environ.get("SOV_KILL_MULT", "2.0"))
KILL_WINDOW = 2
TAU = 1.0
W_CLIP = (0.2, 8.0)
SURVIVE_BONUS = 2.0
WIN_BONUS = 5.0
WIN_SPIKE = 3.0
GATE_SIZE = 64


# --------------------------------------------------------------------------
# Episode bookkeeping
# --------------------------------------------------------------------------

def episode_bounds(lens: Sequence[int]) -> np.ndarray:
    """Cumulative start index of each episode, length len(lens)+1."""
    return np.concatenate([[0], np.cumsum(lens)])


def episode_of(lens: Sequence[int], idx: np.ndarray) -> np.ndarray:
    bounds = episode_bounds(lens)
    return np.searchsorted(bounds[1:], idx, side="right")


# --------------------------------------------------------------------------
# Returns, RTG, weights
# --------------------------------------------------------------------------

def compute_returns(reward: np.ndarray, alive_ep: np.ndarray,
                    lens: Sequence[int], gamma: float = GAMMA,
                    survive_bonus: float = SURVIVE_BONUS,
                    win_bonus: float = WIN_BONUS,
                    win_spike: float = WIN_SPIKE) -> np.ndarray:
    """Return-to-go per frame, then z-scored per shard (shaping-robust).

    2026-08-14 pipeline-agent fix: real shards store `alive` PER EPISODE
    (int 0/1), not per frame — signature is alive_ep (E,).

    Terminal handling per episode:
      - died     -> no bonus
      - survived -> +survive_bonus on the last frame
      - likely win (last reward >= win_spike) -> +win_bonus
    """
    R = np.zeros_like(reward, dtype=np.float32)
    bounds = episode_bounds(lens)
    for e in range(len(lens)):
        s, t = bounds[e], bounds[e + 1]
        if t <= s:
            continue
        r = reward[s:t].astype(np.float32)
        if bool(alive_ep[e]):
            r = r.copy()
            r[-1] += win_bonus if r[-1] >= win_spike else survive_bonus
        acc = 0.0
        for i in range(t - 1, s - 1, -1):
            acc = r[i - s] + gamma * acc
            R[i] = acc
    mu, sd = R.mean(), R.std()
    if sd < 1e-6:
        return np.zeros_like(R)
    return (R - mu) / max(sd, 1e-6)


def shift_within_episode(v: np.ndarray, lens: Sequence[int]) -> np.ndarray:
    """v[t] <- v[t+1] within each episode (last frame repeats)."""
    out = v.copy()
    bounds = episode_bounds(lens)
    for e in range(len(lens)):
        s, t = bounds[e], bounds[e + 1]
        if t > s:
            out[s:t - 1] = v[s + 1:t]
    return out


def frame_weights(Rz: np.ndarray, alive_ep: np.ndarray, lens: Sequence[int],
                  kills: Optional[np.ndarray] = None,
                  tau: float = TAU, clip: Tuple[float, float] = W_CLIP,
                  survivor_mult: float = SURVIVOR_MULT,
                  kill_mult: float = KILL_MULT,
                  window: int = KILL_WINDOW) -> np.ndarray:
    """AWR-style soft filtering weights, normalized to mean 1.
    2026-08-14 fix: alive_ep is PER EPISODE (E,), not per frame."""
    w = np.clip(np.exp(Rz / tau), clip[0], clip[1]).astype(np.float32)
    bounds = episode_bounds(lens)
    for e in range(len(lens)):
        s, t = bounds[e], bounds[e + 1]
        if t > s and bool(alive_ep[e]):
            w[s:t] *= survivor_mult
    if kills is not None:
        dk = np.diff(kills, prepend=kills[:1] - 0.0)
        hit = np.where(dk > 1e-3)[0]
        for i in hit:
            lo, hi = max(0, i - window), min(len(w), i + window + 1)
            w[lo:hi] *= kill_mult
    return w / (w.mean() + 1e-8)


def win_labels(alive_ep: np.ndarray, lens: Sequence[int]) -> np.ndarray:
    """1 if the episode survived to timeout, 0 if it died.
    2026-08-14 fix: alive_ep is PER EPISODE (E,)."""
    lab = np.zeros(int(np.sum(lens)), dtype=np.int64)
    bounds = episode_bounds(lens)
    for e in range(len(lens)):
        s, t = bounds[e], bounds[e + 1]
        if t > s and bool(alive_ep[e]):
            lab[s:t] = 1
    return lab


# --------------------------------------------------------------------------
# Label-derived supervision (gate / seg / dynamics / intent mirrors)
# --------------------------------------------------------------------------

def _downscale_labels(lab: np.ndarray, size: int) -> np.ndarray:
    """(N,H,W) int labels -> (N,size,size) majority-vote labels."""
    N, H, W = lab.shape
    gh, gw = H // size, W // size
    out = np.zeros((N, size, size), dtype=np.int16)
    for gy in range(size):
        for gx in range(size):
            blk = lab[:, gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw]
            flat = blk.reshape(N, -1)
            # majority vote via bincount along axis 1
            vals = np.argmax(
                np.apply_along_axis(lambda r: np.bincount(r, minlength=4),
                                    1, flat.astype(np.int64)), axis=1)
            out[:, gy, gx] = vals
    return out


def gate_masks_from_lab(lab64: np.ndarray, adj: int = 2
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-kind mask targets at 64x64 for kind 0..2 from downsampled labels.

    Returns (masks (N,3,64,64) float in [0,1], valid (N,3) bool).
    Kind 2 (bank) has no spatial target -> flagged invalid.
    """
    N, S, _ = lab64.shape
    masks = np.zeros((N, 3, S, S), dtype=np.float32)
    valid = np.zeros((N, 3), dtype=bool)
    valid[:, 0] = True
    valid[:, 1] = True
    me = lab64 == 2
    enemy = lab64 == 3
    neutral = lab64 == 1
    # expand mask: dilate me by adj and intersect neutral
    pad = adj
    mp = np.pad(me, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    d = np.zeros_like(me)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            d |= mp[:, pad + dy:pad + dy + S, pad + dx:pad + dx + S]
    masks[:, 0] = (d & neutral).astype(np.float32)
    masks[:, 1] = enemy.astype(np.float32)
    return masks, valid


def intent_mirrors(lab64: np.ndarray, lab64_next: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """threat = pixels that BECOME enemy next frame; expand = BECOME mine."""
    threat = ((lab64_next == 3) & (lab64 != 3)).astype(np.float32)
    expand = ((lab64 == 1) & (lab64_next == 2)).astype(np.float32)
    return threat, expand


# --------------------------------------------------------------------------
# Frame stacks + arena filter
# --------------------------------------------------------------------------

def build_stacks(rgb: np.ndarray, lens: Sequence[int],
                 gap: int = 2, K: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """rgb (N,H,W,3) -> (N,3K,H,W) gap-spaced stacks (memory ablation)."""
    N, H, W, C = rgb.shape
    out = np.zeros((N, C * K, H, W), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    rgb_f = rgb.astype(np.float32) / 255.0
    bounds = episode_bounds(lens)
    eps = episode_of(lens, np.arange(N))
    for i in range(N):
        s = bounds[eps[i]]
        ch = []
        ok = True
        for k in range(K):
            j = i - k * gap
            if j < s:
                j = s
                ok = False
            ch.append(rgb_f[j].transpose(2, 0, 1))
        out[i] = np.concatenate(ch, axis=0)
        valid[i] = ok
    return out, valid


def arena_eps_of(lab: np.ndarray, lens: Sequence[int],
                 water_frac: float = 0.85) -> np.ndarray:
    """Per-episode True if the episode looks like the leaky arena map."""
    bounds = episode_bounds(lens)
    flags = []
    for e in range(len(lens)):
        s, t = bounds[e], bounds[e + 1]
        frac = float((lab[s:t] == 0).mean()) if t > s else 0.0
        flags.append(frac > water_frac)
    return np.array(flags, dtype=bool)


# --------------------------------------------------------------------------
# Sequence windows (GRU memory-core training)
# --------------------------------------------------------------------------

def build_sequences(lens: Sequence[int], T: int = 8,
                    stride: int = 4) -> List[Tuple[int, int]]:
    """Return (start, length=T) windows fully inside episodes."""
    bounds = episode_bounds(lens)
    windows: List[Tuple[int, int]] = []
    for e in range(len(lens)):
        s, t = bounds[e], bounds[e + 1]
        i = s
        while i + T <= t:
            windows.append((i, T))
            i += stride
    return windows


# --------------------------------------------------------------------------
# Prepper + Dataset
# --------------------------------------------------------------------------

@dataclass
class PreppedShard:
    rgb: np.ndarray            # (N,3,256,256) float32 in [0,1]
    rgb_next: np.ndarray       # (N,3,256,256) consecutive frame — dynamics
    reward: np.ndarray         # (N,) farmer reward — dynamics reward target
    nums: np.ndarray           # (N,8)
    rtg: np.ndarray            # (N,1) z-scored return conditioning
    rtg_next: np.ndarray       # (N,1)
    nums_next: np.ndarray      # (N,9) cat(nums[t+1], rtg[t+1]) — econ target
    kind: np.ndarray           # (N,)
    cell: np.ndarray           # (N,)
    pct: np.ndarray            # (N,)
    ret: np.ndarray            # (N,) z-scored returns (value target)
    w: np.ndarray              # (N,) sample weights (mean 1)
    win_lab: np.ndarray        # (N,) 0/1
    lab64: np.ndarray          # (N,64,64) int64
    lab64_next: np.ndarray     # (N,64,64) int64 (dynamics target)
    threat: np.ndarray         # (N,64,64) float
    expand: np.ndarray         # (N,64,64) float
    gate_mask: np.ndarray      # (N,3,64,64)
    gate_valid: np.ndarray     # (N,3) bool


class ShardPrepper:
    """Loads + preps one npz shard. Cache the result per epoch."""

    def __init__(self, path: Path, stack: Optional[Tuple[int, int]] = None,
                 skip_arena: bool = True):
        self.path = Path(path)
        self.stack = stack
        self.skip_arena = skip_arena

    @staticmethod
    def _arena_flags(lab: np.ndarray, lens_frames: Sequence[int],
                     lens_ticks: Sequence[int]) -> np.ndarray:
        """Per-episode True = lobby-screenshot arena map (DROP).
        2026-08-14 pipeline-agent: port the old teacher's MEASURED arena
        detector (audit_data.arena_eps_of, watermark-signature based) —
        strictly better than the crude >85%-water heuristic. Fallback to
        the heuristic if audit_data is not importable."""
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from audit_data import arena_eps_of as strict_ae
            return np.asarray(strict_ae(lab, lens_ticks), dtype=bool)
        except Exception:
            return arena_eps_of(lab, lens_frames)

    def prepare(self) -> Optional[PreppedShard]:
        d = np.load(self.path, allow_pickle=True)
        rgb, lab = d["rgb"], d["lab"]
        nums = d["nums"].astype(np.float32)
        kind = d["kind"].astype(np.int64)
        cell = d["cell"].astype(np.int64)
        pct = d["pct"].astype(np.float32)
        reward_raw = d["reward"].astype(np.float32)
        alive_raw = d["alive"]
        lens_raw = [int(l) for l in d["lens"]]
        N = len(rgb)

        # ==== 2026-08-14 pipeline-agent fix — REAL shard layout ============
        # Verified on shard_v2_1786707781 (rgb 354 frames / reward 708):
        #   rgb/lab/nums/kind/cell/pct/logp : per recorded FRAME (REC_EVERY=2)
        #   reward : per TICK  (len == sum(lens))
        #   alive  : PER EPISODE int 0/1 (len == E)
        #   lens   : per-episode TICK counts -> frames = lens // REC_EVERY
        # The original code assumed all per-frame -> IndexError on real data.
        REC_EVERY = 2
        # 17 Aug fix: ODD tick counts (999 ticks -> 500 frames) broke the
        # floor-based detection and fell into the legacy branch -> crash.
        # Ceil per episode: (l+1)//2.
        rec = [max(1, (l + 1) // REC_EVERY) for l in lens_raw]
        if (len(alive_raw) == len(lens_raw)
                and int(np.sum(lens_raw)) == len(reward_raw)
                and int(np.sum(rec)) == N):
            alive_ep = np.asarray(alive_raw).astype(bool)
            lens = rec
            reward = np.zeros(N, dtype=np.float32)   # ticks -> frames (sum)
            off_s = 0
            off_f = 0
            for l_t, r_f in zip(lens_raw, rec):
                for f in range(r_f):
                    lo = off_s + f * REC_EVERY
                    hi = min(off_s + l_t, lo + REC_EVERY)
                    reward[off_f + f] = reward_raw[lo:hi].sum()
                off_s += l_t
                off_f += r_f
        else:
            # synthetic / legacy path: lens in frames, alive per frame
            lens = lens_raw
            reward = reward_raw
            alive_f = np.asarray(alive_raw).astype(bool)
            bounds_ = episode_bounds(lens)
            alive_ep = np.array(
                [bool(alive_f[bounds_[e + 1] - 1]) for e in range(len(lens))])
        # =====================================================================

        if self.skip_arena:
            ae = self._arena_flags(lab, lens, lens_raw)
            if ae.all():
                return None

        if self.stack is not None:
            gap, K = self.stack
            rgb_f, _ok = build_stacks(rgb, lens, gap=gap, K=K)
        else:
            rgb_f = rgb.astype(np.float32).transpose(0, 3, 1, 2) / 255.0

        # returns / rtg / weights / win labels (per-episode alive, 2026-08-14)
        ret = compute_returns(reward, alive_ep, lens)
        rtg = ret[:, None].astype(np.float32)
        rtg_next = shift_within_episode(rtg, lens)
        nums_next8 = shift_within_episode(nums, lens)
        nums_next = np.concatenate([nums_next8, rtg_next], axis=1)  # (N,9)
        w = frame_weights(ret, alive_ep, lens, kills=nums[:, 7])
        win_lab = win_labels(alive_ep, lens)

        # label-derived supervision at 64x64
        lab64 = _downscale_labels(lab, GATE_SIZE).astype(np.int64)
        lab_next = np.empty_like(lab)
        lab_next[:-1] = lab[1:]
        lab_next[-1] = lab[-1]
        # fix episode boundaries: last frame of an episode repeats itself
        bounds = episode_bounds(lens)
        for e in range(len(lens)):
            t = bounds[e + 1]
            if t < len(lab):
                lab_next[t - 1] = lab[t - 1]
        lab64_next = _downscale_labels(lab_next, GATE_SIZE).astype(np.int64)
        threat, expand = intent_mirrors(lab64, lab64_next)
        gmask, gvalid = gate_masks_from_lab(lab64)

        keep = np.ones(len(rgb_f), dtype=bool)
        if self.skip_arena:
            eps = episode_of(lens, np.arange(len(rgb_f)))
            keep = ~ae[eps]
        if not keep.any():
            return None

        # consecutive-frame dynamics targets (last frame of each episode
        # repeats itself — no cross-episode leakage)
        rgb_next = rgb_f.copy()
        for e in range(len(lens)):
            s, t = bounds[e], bounds[e + 1]
            if t > s:
                rgb_next[s:t - 1] = rgb_f[s + 1:t]

        return PreppedShard(
            rgb=rgb_f[keep], rgb_next=rgb_next[keep], reward=reward[keep],
            nums=nums[keep], rtg=rtg[keep],
            rtg_next=rtg_next[keep], nums_next=nums_next[keep],
            kind=kind[keep], cell=cell[keep], pct=pct[keep], ret=ret[keep],
            w=w[keep], win_lab=win_lab[keep], lab64=lab64[keep],
            lab64_next=lab64_next[keep], threat=threat[keep],
            expand=expand[keep], gate_mask=gmask[keep],
            gate_valid=gvalid[keep])


class StageADataset(Dataset):
    """Concatenated prepped shards; sampled with WeightedRandomSampler."""

    def __init__(self, prepped: Sequence[PreppedShard]):
        def cat(key):
            return np.concatenate([getattr(p, key) for p in prepped])
        self.rgb = cat("rgb")
        self.rgb_next = cat("rgb_next")
        self.reward = cat("reward")
        self.nums = cat("nums")
        self.rtg = cat("rtg")
        self.rtg_next = cat("rtg_next")
        self.nums_next = cat("nums_next")
        self.kind = cat("kind")
        self.cell = cat("cell")
        self.pct = cat("pct")
        self.ret = cat("ret")
        self.w = cat("w")
        self.win_lab = cat("win_lab")
        self.lab64 = cat("lab64")
        self.lab64_next = cat("lab64_next")
        self.threat = cat("threat")
        self.expand = cat("expand")
        self.gmask = cat("gate_mask")
        self.gvalid = cat("gate_valid")

    def __len__(self) -> int:
        return len(self.rgb)

    def __getitem__(self, i: int) -> dict:
        t = torch.from_numpy
        return dict(
            rgb=t(self.rgb[i]), rgb_next=t(self.rgb_next[i]),
            # 2026-08-14 fix: from_numpy rejects 0-dim scalars
            reward=torch.tensor(self.reward[i]), nums=t(self.nums[i]),
            rtg=t(self.rtg[i]),
            rtg_next=t(self.rtg_next[i]), nums_next=t(self.nums_next[i]),
            kind=torch.tensor(self.kind[i]), cell=torch.tensor(self.cell[i]),
            pct=torch.tensor(self.pct[i]), ret=torch.tensor(self.ret[i]),
            w=torch.tensor(self.w[i]), win_lab=torch.tensor(self.win_lab[i]),
            lab64=t(self.lab64[i]), lab64_next=t(self.lab64_next[i]),
            threat=t(self.threat[i]), expand=t(self.expand[i]),
            gate_mask=t(self.gmask[i]), gate_valid=t(self.gvalid[i]))


def make_stage_a_loader(shard_paths: Sequence[Path], batch_size: int = 16,
                        num_workers: int = 4, shuffle_shards: bool = True,
                        stack: Optional[Tuple[int, int]] = None,
                        seed: int = 0) -> Iterator[dict]:
    """Shard-at-a-time prepping + weighted sampling (memory-efficient)."""
    paths = list(shard_paths)
    rng = np.random.default_rng(seed)
    if shuffle_shards:
        rng.shuffle(paths)
    for sp in paths:
        prepped = ShardPrepper(sp, stack=stack).prepare()
        if prepped is None:
            continue
        ds = StageADataset([prepped])
        sampler = WeightedRandomSampler(ds.w, num_samples=len(ds) // 2,
                                        replacement=True)
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, drop_last=True)
        for batch in dl:
            yield batch


if __name__ == "__main__":
    # quick sanity run on fake data
    rng = np.random.default_rng(0)
    N, lens = 120, [40, 40, 40]
    lab = np.zeros((N, 256, 256), dtype=np.uint8)
    lab[:, 10:200, 10:200] = 1            # neutral
    lab[:, 20:80, 20:80] = 2              # me
    lab[:, 100:180, 100:180] = 3          # enemy
    lab_next = lab.copy()
    lab_next[:, 180:190, 100:180] = 3     # enemy advances (threat)
    lab_next[:, 80:90, 20:80] = 2         # I expand
    rgb = (lab[:, :, :, None].astype(np.float32) * 40).astype(np.uint8)
    rgb = np.concatenate([rgb] * 3, axis=-1)
    alive = np.ones(N, dtype=bool)
    alive[75] = False
    # per-episode outcomes (real-shard contract, 2026-08-14): episode 0 died
    # (alive[75]=False inside frames 0..39), episodes 1,2 survived
    alive_ep = np.array([False, True, True])
    ret = compute_returns(np.zeros(N, dtype=np.float32), alive_ep, lens)
    w = frame_weights(ret, alive_ep, lens, kills=np.zeros(N))
    lab64 = _downscale_labels(lab, 64)
    lab64_next = _downscale_labels(lab_next, 64)
    threat, expand = intent_mirrors(lab64, lab64_next)
    masks, valid = gate_masks_from_lab(lab64)
    print("returns z-range:", ret.min(), ret.max())
    print("weights mean:", w.mean(), "max:", w.max())
    print("threat frac:", threat.mean(), "expand frac:", expand.mean())
    print("masks:", masks.shape, "valid per kind:", valid.sum(0))
    print("windows (T=8, stride=4):", len(build_sequences(lens, 8, 4)))
    print("win labels:", np.bincount(win_labels(alive_ep, lens)))
    print("adapter smoke OK")
