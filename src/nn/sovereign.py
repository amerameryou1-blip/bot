"""
SOVEREIGN v3 — ~290M-param return-conditioned spatial policy with MCTS
planning for territorial.io (last-survivor elimination).

= DESIGN SUMMARY (full reasoning: ARCHITECTURE.md, training: train_notes.md)
Size mandate: >=200M, <400M (user-specified). Four modules:

    VISION SPINE      ~70M   5 down stages 96..832ch @ 64..4 px, ResBlock+SE
    STRATEGIC CORTEX  ~77M   multi-scale transformer: 81 tokens (1 GLOBAL +
                             16 coarse @4px + 64 fine @8px), 6 layers,
                             d=1024, FFN=4096, 16 heads
    TACTICAL DECODER  ~66M   6 up stages w/ skips, 1024ch @8px -> 64ch @256px
    COMMAND HEADS     ~59M   kind/value/win/econ 3072-wide MLPs, heat towers,
                             pct-Beta, gate/seg/dynamics aux, GRU memory

= THE 10x LEVERS OVER v2 (each grounded in literature / free labels)
  1. RETURN-CONDITIONED POLICY (RPO, ICLR'23): all heads are conditioned on
     remaining return-to-go `rtg`. Stage A teaches p(a|s,rtg); PPO computes
     advantages vs V(s,rtg); at play time set an aspiration rtg -> the
     policy *demands* winning. Default rtg=0 keeps the contract.
  2. OPPONENT-INTENT MIRROR HEADS: shards are consecutive frames, so
     lab[t+1] is free supervision. `threat` predicts where enemies strike
     next; `expand` predicts where my frontier grows. This is implicit
     opponent modeling from data (TWISTER/Dreamer-class prediction aux).
  3. ONE-STEP SPATIAL DYNAMICS: `next_seg` predicts lab[t+1] @64x64 — a
     partial world model in the policy (seed for future MCTS).
  4. WIN-PROBABILITY VALUE: second value head outputs P(alive at end).
     GAE uses the RTG value; win-prob drives eval + late-game shaping.
  5. MEMORY CORE (GRU 768): optional recurrent state over the cortex pool
     (contract-safe default off; truncated-BPTT recipe documented).
  6. IQL-STYLE EXPECTILE VALUE option in Stage A for robustness on the
     weak farmer buffer, next to the AWR-weighted imitation.
  7. LEAGUE-LITE snapshots: PPO trains against frozen past selves + bots
     in the sim (fictitious self-play lite), scheduled in train_notes.
  8. WAKE-SLEEP distill loop: PPO on the big net -> distill into a small
     student for faster rollouts -> retrain. Documented recipe.

= CONTRACT (kept verbatim, see README.md)
    INPUT : rgb (B,3,256,256) float [0,1] + nums (B,8)  [+ rtg (B,) + state]
    OUTPUT: click heatmap >=16x16 (native 256x256, pooled via grid=),
            kind logits (B,3), pct in [0,1], value (B,)
    `lab` is supervision-only during pretraining; never a play-time input.

= MEMORY CHECK (2xT4 16GB, fp16, bs16/GPU): weights 1.09GB + grads 1.09GB +
  Adam 2.18GB + activations ~4GB + ctx ~1GB ~= 9.4GB/GPU. Optional gradient
  checkpointing on the 4 biggest stages (flag `grad_ckpt`) halves activation
  cost if needed. Play speed ~150-200 GFLOPs/frame < 30ms vs 560ms ticks.

torch only. Run `python model.py` (CPU smoke: params + shapes),
`python model.py --fast` for the 128px path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
from torch.utils.checkpoint import checkpoint as _ckpt

EPS = 1e-6


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class SovereignConfig:
    map_size: int = 256
    in_c: int = 3                # contract default: single rgb frame
    ctx_dim: int = 8             # nums vector
    rtg_dim: int = 1             # return-conditioning channel (0 disables)
    num_kinds: int = 3           # 0 expand / 1 attack / 2 bank
    # vision spine: enc_ch[0]=stem out; enc_ch[1..5]=stage outs; blocks=res
    enc_ch: Tuple[int, ...] = (96, 96, 160, 288, 512, 832)
    enc_blocks: Tuple[int, ...] = (2, 3, 4, 4, 3)
    # cortex
    cortex_d: int = 1024
    cortex_heads: int = 16
    cortex_ffn: int = 4096
    cortex_layers: int = 6
    cortex_max_tokens: int = 512   # 1 global + 16 coarse + 64 fine = 81
    # decoder: (out_ch, skip_ch or None, res_blocks) — input starts at
    # enc_ch[-1]; dec5 output channel == dec_out_ch
    dec_plan: Tuple[Tuple[int, Optional[int], int], ...] = (
        (1024, 512, 2),   # dec0: 832->1024 @H/32 (skip e4)
        (512, 288, 2),    # dec1 @H/16 (skip e3)
        (288, 160, 2),    # dec2 @H/8  (skip e2)  [d32 -> cell_value]
        (160, 96, 2),     # dec3 @H/4  (skip e1)  [d64 -> gate/seg sources]
        (96, None, 1),    # dec4 @H/2
        (64, None, 1),    # dec5 @H    (dec_out_ch = 64)
    )
    dec_out_ch: int = 64          # must equal dec_plan[-1][0]
    heat_hidden: int = 128        # shared heat mixer before per-kind towers
    tower_hidden: int = 192       # per-kind tower width
    head_hidden: int = 3072       # kind/value/win/econ MLP width
    pct_hidden: int = 1024        # Beta head width
    grid_default: int = 32
    # memory core
    use_mem: bool = True          # GRU over cortex pool + context
    mem_hidden: int = 768
    # features
    use_coord: bool = True
    use_gate: bool = True
    use_aux_seg: bool = True
    use_dynamics: bool = True     # next_seg / threat / expand aux heads
    use_econ: bool = True         # next-context forecaster
    use_cell_value: bool = True   # per-cell utility grid @32x32
    use_se: bool = True           # squeeze-excite channel attention
    aux_seg_classes: int = 4      # water / neutral / me / enemy
    dropout_mlp: float = 0.1
    grad_ckpt: bool = False       # checkpoint the 4 biggest stages
    # -- planning / search block (MuZero-lite) ------------------------------
    use_search: bool = True       # latent encode/predict/dynamics for MCTS
    lat_c: int = 32               # compressed spatial channels in the latent
    lat_spatial: int = 8          # latent map size (8x8)
    dyn_hidden: int = 1024        # dynamics MLP hidden width
    act_embed: int = 512          # action embedding width

    @property
    def ctx_full_dim(self) -> int:
        return self.ctx_dim + self.rtg_dim

    @property
    def latent_dim(self) -> int:
        return self.cortex_d + self.lat_c * self.lat_spatial * self.lat_spatial


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------

class _SE(nn.Module):
    """Squeeze-excite channel attention."""

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        h = max(c // r, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(c, h, 1, bias=False), nn.SiLU(),
            nn.Conv2d(h, c, 1, bias=False), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(F.adaptive_avg_pool2d(x, 1))


class ResBlock(nn.Module):
    """conv3x3-BN-SiLU, conv3x3-BN, (+SE), residual."""

    def __init__(self, c: int, se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.se = _SE(c) if se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.silu(x + self.se(y))


class DownStage(nn.Module):
    """stride-2 conv + n residual blocks."""

    def __init__(self, cin: int, cout: int, blocks: int, se: bool = True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(cout), nn.SiLU()]
        layers += [ResBlock(cout, se) for _ in range(blocks)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpStage(nn.Module):
    """bilinear x2 + skip merge + n residual blocks."""

    def __init__(self, cin: int, cout: int, skip_c: Optional[int] = None,
                 blocks: int = 1, se: bool = True):
        super().__init__()
        self.skip = (nn.Conv2d(skip_c, cin, 1, bias=False)
                     if skip_c is not None else None)
        self.up = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.blocks = nn.Sequential(*[ResBlock(cout, se)
                                      for _ in range(blocks)])

    def forward(self, x: torch.Tensor,
                skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear",
                          align_corners=False)
        if skip is not None:
            x = x + self.skip(skip)
        x = F.silu(self.bn(self.up(x)))
        return self.blocks(x)


class FiLM(nn.Module):
    """Per-channel scale/shift of the bottleneck from ctx (+rtg)."""

    def __init__(self, ctx_dim: int, c: int):
        super().__init__()
        self.gamma = nn.Linear(ctx_dim, c)
        self.beta = nn.Linear(ctx_dim, c)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        g = self.gamma(ctx).view(ctx.shape[0], -1, 1, 1) + 1.0
        b = self.beta(ctx).view(ctx.shape[0], -1, 1, 1)
        return x * g + b


# --------------------------------------------------------------------------
# Multi-scale strategic cortex
# --------------------------------------------------------------------------

class _Attn(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.heads, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.scale = self.dh ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.dh).transpose(1, 2)
        k = k.view(B, T, self.heads, self.dh).transpose(1, 2)
        v = v.view(B, T, self.heads, self.dh).transpose(1, 2)
        a = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        o = (a @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(o)


class _TxBlock(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = _Attn(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Cortex(nn.Module):
    """Multi-scale patch tokens: [GLOBAL] + 16 coarse (@4px) + 64 fine
    (@8px) -> 6 pre-norm attention layers. The FFNs carry the strategic
    capacity; attention over 81 tokens is nearly free compute."""

    def __init__(self, cfg: SovereignConfig):
        super().__init__()
        self.coarse_embed = nn.Conv2d(cfg.enc_ch[-1], cfg.cortex_d, 1,
                                      bias=False)
        self.fine_embed = nn.Conv2d(cfg.enc_ch[-2], cfg.cortex_d, 1,
                                    bias=False)
        self.global_tok = nn.Parameter(torch.zeros(1, 1, cfg.cortex_d))
        self.pos = nn.Parameter(torch.zeros(1, cfg.cortex_max_tokens,
                                            cfg.cortex_d))
        nn.init.trunc_normal_(self.global_tok, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            _TxBlock(cfg.cortex_d, cfg.cortex_heads, cfg.cortex_ffn,
                     dropout=cfg.dropout_mlp)
            for _ in range(cfg.cortex_layers)])
        self.ln_f = nn.LayerNorm(cfg.cortex_d)

    def forward(self, bot: torch.Tensor,
                fine: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = bot.shape[0]
        t_c = self.coarse_embed(bot).flatten(2).transpose(1, 2)   # (B,16,d)
        t_f = self.fine_embed(fine).flatten(2).transpose(1, 2)    # (B,64,d)
        g = self.global_tok.expand(B, -1, -1)
        t = torch.cat([g, t_c, t_f], dim=1)                       # (B,81,d)
        t = t + self.pos[:, :t.shape[1]]
        for blk in self.blocks:
            t = blk(t)
        t = self.ln_f(t)
        return t, t[:, 0]      # tokens, global


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, cin: int, hidden: int, cout: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cin, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BetaHead(nn.Module):
    """pct in [0,1] as Beta(alpha,beta); alpha,beta >= 1 via softplus+1."""

    def __init__(self, in_c: int, ctx_dim: int, num_kinds: int,
                 hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_c + ctx_dim + num_kinds, hidden),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2))

    def forward(self, feat: torch.Tensor, ctx: torch.Tensor,
                kind_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([feat, ctx, kind_onehot], dim=1)
        a, b = self.net(h).chunk(2, dim=1)
        return F.softplus(a) + 1.0, F.softplus(b) + 1.0


class _ConvBn(nn.Module):
    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class _KindTower(nn.Module):
    """Per-kind heat tower (one policy per action kind)."""

    def __init__(self, cin: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBn(cin, hidden, 3), _ConvBn(hidden, hidden, 3),
            nn.Conv2d(hidden, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)        # (B,H,W)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Sovereign(nn.Module):
    """SOVEREIGN v3 — see module docstring for the contract."""

    def __init__(self, cfg: Optional[SovereignConfig] = None):
        super().__init__()
        self.cfg = cfg or SovereignConfig()
        c = self.cfg
        H = c.map_size
        assert c.dec_plan[-1][0] == c.dec_out_ch, \
            "dec_plan[-1][0] must equal dec_out_ch"

        # -- stem -----------------------------------------------------------
        in_c = c.in_c + (2 if c.use_coord else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, c.enc_ch[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c.enc_ch[0]), nn.SiLU())
        if c.use_coord:
            ys = torch.linspace(-1, 1, H).view(1, 1, H, 1).repeat(1, 1, 1, H)
            xs = torch.linspace(-1, 1, H).view(1, 1, 1, H).repeat(1, 1, H, 1)
            self.register_buffer("coords", torch.cat([ys, xs], dim=1))

        # -- vision spine ---------------------------------------------------
        ch = c.enc_ch[0]
        self.enc = nn.ModuleList()
        for i, blocks in enumerate(c.enc_blocks):
            self.enc.append(DownStage(ch, c.enc_ch[i + 1], blocks, c.use_se))
            ch = c.enc_ch[i + 1]

        self.film = FiLM(c.ctx_full_dim, ch)     # on bottleneck (e5)
        self.cortex = Cortex(c)

        # -- decoder --------------------------------------------------------
        self.dec = nn.ModuleList()
        for cout, skip_c, blocks in c.dec_plan:
            self.dec.append(UpStage(ch, cout, skip_c, blocks, c.use_se))
            ch = cout

        # -- heat heads ------------------------------------------------------
        self.heat_mixer = nn.Sequential(
            _ConvBn(c.dec_out_ch, c.heat_hidden, 3),
            _ConvBn(c.heat_hidden, c.heat_hidden, 3))
        self.kind_towers = nn.ModuleList([
            _KindTower(c.heat_hidden, c.tower_hidden)
            for _ in range(c.num_kinds)])

        # -- spatial aux heads (on e1 @H/4) ---------------------------------
        if c.use_gate:
            self.gate_head = nn.Conv2d(c.enc_ch[1], c.num_kinds, 1)
        if c.use_aux_seg:
            self.seg_head = nn.Conv2d(c.enc_ch[1], c.aux_seg_classes, 1)
        if c.use_dynamics:
            self.next_seg_head = nn.Conv2d(c.enc_ch[1], c.aux_seg_classes, 1)
            self.threat_head = nn.Conv2d(c.enc_ch[1], 1, 1)
            self.expand_head = nn.Conv2d(c.enc_ch[1], 1, 1)

        # -- global heads ----------------------------------------------------
        pool_dim = c.cortex_d + c.ctx_full_dim
        self.kind_head = _MLP(pool_dim, c.head_hidden, c.num_kinds,
                              c.dropout_mlp)
        self.value_head = _MLP(pool_dim, c.head_hidden, 1, c.dropout_mlp)
        self.win_head = _MLP(pool_dim, c.head_hidden, 2, c.dropout_mlp)
        self.econ_head = (_MLP(pool_dim, c.head_hidden, c.ctx_full_dim,
                               c.dropout_mlp) if c.use_econ else None)
        self.pct_head = BetaHead(c.dec_out_ch, c.ctx_full_dim, c.num_kinds,
                                 hidden=c.pct_hidden, dropout=c.dropout_mlp)
        self.cell_value = (
            nn.Sequential(_ConvBn(c.dec_plan[2][0], 128, 1),
                          nn.Conv2d(128, 1, 1))
            if c.use_cell_value else None)

        # -- memory core -----------------------------------------------------
        self.mem = None
        if c.use_mem:
            self.mem = nn.GRU(pool_dim, c.mem_hidden, 1, batch_first=True)
            self.mem_proj = nn.Linear(c.mem_hidden, c.cortex_d)

        # -- planning block (MuZero-lite): latent encode/predict/dynamics ---
        # Trained on CONSECUTIVE shard frames (rgb_t, a_t -> rgb_{t+1}):
        #   encode   -> z  (B, latent_dim)
        #   prediction(z) -> prior logits over (kind,cell), value, win
        #   dynamics(z, a) -> z_next, reward_hat
        # Drives MCTS at decision time (see search.py). ~17M params.
        self.planning = None
        if c.use_search:
            m = nn.Module()
            m.lat_proj = nn.Sequential(
                nn.Conv2d(c.enc_ch[-1], c.lat_c, 1, bias=False),
                nn.BatchNorm2d(c.lat_c), nn.SiLU())
            m.a_emb_kind = nn.Embedding(c.num_kinds, 64)
            m.a_emb_cell = nn.Embedding(c.grid_default * c.grid_default, 256)
            m.a_emb_pct = nn.Linear(1, 128)
            m.a_emb_out = nn.Linear(64 + 256 + 128, c.act_embed)
            m.dyn_mlp = nn.Sequential(
                nn.Linear(c.latent_dim + c.act_embed, c.dyn_hidden),
                nn.SiLU(), nn.Linear(c.dyn_hidden, c.latent_dim))
            m.reward_head = nn.Linear(c.latent_dim, 1)
            m.pred_prior = nn.Linear(
                c.latent_dim, c.num_kinds * c.grid_default * c.grid_default)
            m.pred_value = nn.Linear(c.latent_dim, 1)
            m.pred_win = nn.Linear(c.latent_dim, 2)
            self.planning = m

        self.apply(_init_weights)

    # -- helpers ------------------------------------------------------------

    def _staged(self, m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Optional gradient checkpointing on the heavy stages."""
        if self.cfg.grad_ckpt and torch.is_grad_enabled():
            return _ckpt(m, x, use_reentrant=False)
        return m(x)

    @staticmethod
    def _coord_norm(yx: torch.Tensor, size: int) -> torch.Tensor:
        return (yx.float() / max(size - 1, 1)) * 2.0 - 1.0

    @staticmethod
    def _gather(feat: torch.Tensor, yx: torch.Tensor) -> torch.Tensor:
        coords = Sovereign._coord_norm(yx, feat.shape[-1]).flip(-1)
        return F.grid_sample(feat, coords.view(-1, 1, 1, 2),
                             align_corners=False).squeeze(-1).squeeze(-1)

    def _ctx_full(self, nums: torch.Tensor,
                  rtg: Optional[torch.Tensor]) -> torch.Tensor:
        B = nums.shape[0]
        if rtg is not None:
            r = rtg.view(B, -1).float()
        else:
            r = torch.zeros(B, self.cfg.rtg_dim, device=nums.device)
        return torch.cat([nums, r], dim=1)

    def _gate_logit(self, feat_q: torch.Tensor, H: int) -> torch.Tensor:
        g = torch.sigmoid(self.gate_head(feat_q))
        logg = torch.log(g.clamp_min(EPS))
        if logg.shape[-1] != H:
            logg = F.interpolate(logg, size=(H, H), mode="bilinear",
                                 align_corners=False)
        return logg

    # -- planning interface (used by search.py) -----------------------------

    def encode(self, rgb: torch.Tensor, nums: Optional[torch.Tensor] = None,
               rtg: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compressed latent state z (B, latent_dim) for MCTS + dynamics
        training. Runs the vision spine + cortex (no decoder)."""
        assert self.planning is not None, "planning block disabled"
        cfg = self.cfg
        B, C, H, W = rgb.shape
        assert H == W == cfg.map_size, \
            f"expected {cfg.map_size}x{cfg.map_size}, got {H}x{W}"
        if nums is None:
            nums = torch.zeros(B, cfg.ctx_dim, device=rgb.device)
        ctx_full = self._ctx_full(nums, rtg)
        x = rgb
        if cfg.use_coord:
            x = torch.cat([x, self.coords.expand(B, -1, -1, -1)], dim=1)
        x = self.stem(x)
        prev = None
        for i, stage in enumerate(self.enc):
            prev = x
            x = self._staged(stage, x) if i >= len(self.enc) - 2 else stage(x)
        # x = e5 output, prev = e4 output (cortex fine tokens)
        bot = self.film(x, ctx_full)
        _, g = self.cortex(bot, prev)
        lat_map = F.adaptive_avg_pool2d(
            self.planning.lat_proj(bot),
            (cfg.lat_spatial, cfg.lat_spatial)).flatten(1)
        return torch.cat([g, lat_map], dim=1)      # (B, latent_dim)

    def prediction(self, z: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MuZero-lite prediction net: prior logits over (kind,cell) joint
        actions (B, 3*g*g), value (B,), win logits (B,2)."""
        assert self.planning is not None, "planning block disabled"
        prior = self.planning.pred_prior(z)
        value = self.planning.pred_value(z).squeeze(-1)
        win = self.planning.pred_win(z)
        return prior, value, win

    def dynamics(self, z: torch.Tensor, kind: torch.Tensor,
                 cell: torch.Tensor,
                 pct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step latent transition: (z, a) -> (z_next, reward_hat)."""
        assert self.planning is not None, "planning block disabled"
        cfg = self.cfg
        B = z.shape[0]
        ka = self.planning.a_emb_kind(kind)                          # (B,64)
        ca = self.planning.a_emb_cell(
            cell.clamp(0, cfg.grid_default * cfg.grid_default - 1))  # (B,256)
        pa = self.planning.a_emb_pct(
            pct.clamp(0.0, 1.0).to(z.dtype).view(B, 1))             # (B,128)
        a = self.planning.a_emb_out(torch.cat([ka, ca, pa], dim=1))  # (B,512)
        dz = self.planning.dyn_mlp(torch.cat([z, a], dim=1))
        z_next = z + dz                                              # residual
        r_hat = self.planning.reward_head(z_next).squeeze(-1)
        return z_next, r_hat

    # -- forward ------------------------------------------------------------

    def forward(self, rgb: torch.Tensor, nums: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                grid: Optional[int] = None,
                cell: Optional[torch.Tensor] = None,
                rtg: Optional[torch.Tensor] = None,
                state: Optional[torch.Tensor] = None,
                return_all: bool = False):
        """Contract forward.

        rgb:   (B, C, H, W) float [0,1], H=W=map_size.
        nums:  (B, 8) context (zeros if None).
        rtg:   (B,) return-to-go conditioning (RPO); zeros if None.
        state: (1, B, mem_hidden) optional GRU state (returned in out dict).
        mask:  (B, H, W) external legality mask, log-summed on top.
        grid:  pooled cell grid for the compat heatmap (default 32).
        cell:  (B,) grid idx — pct params evaluated there (training).
        """
        cfg = self.cfg
        B, C, H, W = rgb.shape
        assert H == W == cfg.map_size, \
            f"expected {cfg.map_size}x{cfg.map_size}, got {H}x{W}"
        if nums is None:
            nums = torch.zeros(B, cfg.ctx_dim, device=rgb.device)
        ctx_full = self._ctx_full(nums, rtg)          # (B, ctx+rtg)

        x = rgb
        if cfg.use_coord:
            x = torch.cat([x, self.coords.expand(B, -1, -1, -1)], dim=1)

        # vision spine
        skips = []
        x = self.stem(x)
        for i, stage in enumerate(self.enc):
            x = self._staged(stage, x) if i >= len(self.enc) - 2 else stage(x)
            skips.append(x)               # [@H/4, H/8, H/16, H/32, H/64]

        # cortex
        bot = self.film(x, ctx_full)
        tokens, g = self.cortex(bot, skips[-2])

        # memory core (optional)
        mem_state = None
        if self.mem is not None:
            xin = torch.cat([g, ctx_full], dim=1).unsqueeze(1)   # (B,1,d')
            h0 = state if state is not None else torch.zeros(
                1, B, cfg.mem_hidden, device=rgb.device)
            o, hn = self.mem(xin, h0)
            g = self.mem_proj(o.squeeze(1))                       # (B,d)
            mem_state = hn

        # decoder
        d = (self._staged(lambda t: self.dec[0](t, skips[-2]), bot)
             if cfg.grad_ckpt else self.dec[0](bot, skips[-2]))   # @H/32
        if cfg.grad_ckpt:
            d = self._staged(
                lambda t: self.dec[1](t, skips[-3]), d)
        else:
            d = self.dec[1](d, skips[-3])    # @H/16
        d32 = self.dec[2](d, skips[-4])      # @H/8
        d64 = self.dec[3](d32, skips[-5])    # @H/4
        d128 = self.dec[4](d64, None)        # @H/2
        dres = self.dec[5](d128, None)       # @H (dec_out_ch)

        # heat: mixer + per-kind towers + gate (+ external mask)
        hmix = self.heat_mixer(dres)
        heat_raw = torch.stack([tower(hmix) for tower in self.kind_towers],
                               dim=1)                            # (B,3,H,W)
        heat = heat_raw
        if cfg.use_gate:
            heat = heat + self._gate_logit(skips[0], H)
        if mask is not None:
            assert mask.shape[-2:] == (H, W)
            heat = heat + torch.log(mask.unsqueeze(1).clamp_min(EPS))

        gd = grid if grid is not None else cfg.grid_default
        heat_pool = F.avg_pool2d(heat, H // gd).view(B, cfg.num_kinds, -1)
        click_marginal = torch.logsumexp(heat_pool, dim=1)      # (B, g*g)

        # global heads
        feat = torch.cat([g, ctx_full], dim=1)                  # (B, d+9)
        kind_logits = self.kind_head(feat)
        value = self.value_head(feat).squeeze(-1)
        win_logits = self.win_head(feat)
        econ = self.econ_head(feat) if self.econ_head is not None else None
        kind_soft = F.softmax(kind_logits, dim=1)

        # pct params at requested cell, else at argmax of the marginal
        if cell is not None:
            y = (cell // gd) * (H // gd) + (H // gd) // 2
            xc = (cell % gd) * (H // gd) + (H // gd) // 2
            yx = torch.stack([y, xc], dim=1).to(rgb.device)
        else:
            best = torch.argmax(click_marginal, dim=1)
            y = (best // gd) * (H // gd) + (H // gd) // 2
            xc = (best % gd) * (H // gd) + (H // gd) // 2
            yx = torch.stack([y, xc], dim=1).to(rgb.device)
        pct_feat = self._gather(dres, yx)
        alpha, beta_p = self.pct_head(pct_feat, ctx_full, kind_soft)
        # 2026-08-14 pipeline-agent fix: pct_head emits (B,1); flatten to (B,)
        # or Beta log_prob/entropy broadcast against (B,) terms into B×B
        # matrices in act()/stage_a_loss at batch>1 (verified bug).
        alpha, beta_p = alpha.view(B), beta_p.view(B)
        pct = alpha / (alpha + beta_p)

        out = dict(
            kind_logits=kind_logits,
            kind_probs=kind_soft,
            heat=heat,
            heat_raw=heat_raw,
            cell_logits=heat_pool,
            click=click_marginal,
            pct=pct,
            pct_params=(alpha, beta_p),
            value=value,
            win_logits=win_logits,
            win_prob=torch.softmax(win_logits, dim=1)[:, 1],
            econ=econ,
            gate=torch.sigmoid(self.gate_head(skips[0])) if cfg.use_gate
            else None,
            seg=self.seg_head(skips[0]) if cfg.use_aux_seg else None,
            next_seg=self.next_seg_head(skips[0]) if cfg.use_dynamics
            else None,
            threat=torch.sigmoid(self.threat_head(skips[0])).squeeze(1)
            if cfg.use_dynamics else None,
            expand=torch.sigmoid(self.expand_head(skips[0])).squeeze(1)
            if cfg.use_dynamics else None,
            cell_value=(self.cell_value(d32).squeeze(1)
                        if self.cell_value is not None else None),
            dec_feat=dres,
            mem_state=mem_state,
        )
        if return_all:
            return out
        return out["click"], kind_logits, pct, value

    # -- acting -------------------------------------------------------------

    @torch.no_grad()
    def act(self, rgb: torch.Tensor, nums: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            grid: Optional[int] = None,
            rtg: Optional[torch.Tensor] = None,
            state: Optional[torch.Tensor] = None,
            deterministic: bool = False, temperature: float = 1.0) -> dict:
        """Sample a full action with the correct factorization:
        p(kind) · p(cell | kind) · Beta(pct | kind, cell). Returns state."""
        cfg = self.cfg
        out = self.forward(rgb, nums, mask=mask, grid=grid, rtg=rtg,
                           state=state, return_all=True)
        B = rgb.shape[0]
        gd = grid if grid is not None else cfg.grid_default
        H = cfg.map_size

        kp = out["kind_probs"] / max(temperature, EPS)
        kd = Categorical(logits=torch.log(kp.clamp_min(EPS)))
        kind = kd.sample() if not deterministic else torch.argmax(kp, dim=1)

        heat_k = out["heat"][torch.arange(B, device=rgb.device), kind]
        heat_pool_k = F.avg_pool2d(heat_k, H // gd).view(B, -1)
        cd = Categorical(logits=heat_pool_k / max(temperature, EPS))
        cell = cd.sample() if not deterministic else torch.argmax(
            heat_pool_k, dim=1)

        y = (cell // gd) * (H // gd) + (H // gd) // 2
        xc = (cell % gd) * (H // gd) + (H // gd) // 2
        yx = torch.stack([y, xc], dim=1).to(rgb.device)
        pct_feat = self._gather(out["dec_feat"], yx)
        kind_oh = F.one_hot(kind, cfg.num_kinds).float()
        ctx_full = self._ctx_full(
            nums if nums is not None
            else torch.zeros(B, cfg.ctx_dim, device=rgb.device), rtg)
        alpha, beta_p = self.pct_head(pct_feat, ctx_full, kind_oh)
        alpha, beta_p = alpha.view(B), beta_p.view(B)   # 2026-08-14 fix (see forward)
        beta_d = Beta(alpha, beta_p)
        pct = beta_d.sample() if not deterministic else beta_d.mean

        logprob = (kd.log_prob(kind) + cd.log_prob(cell)
                   + beta_d.log_prob(pct.clamp(EPS, 1 - EPS)))
        entropy = (kd.entropy() + cd.entropy() + beta_d.entropy())
        return dict(kind=kind, cell=cell, yx=yx, pct=pct, logprob=logprob,
                    value=out["value"], win_prob=out["win_prob"],
                    entropy=entropy, kind_probs=out["kind_probs"],
                    state=out["mem_state"])


# --------------------------------------------------------------------------
# Stage-A (offline pretraining) loss — used with data_adapters.py
# --------------------------------------------------------------------------

def stage_a_loss(out: dict, kind: torch.Tensor, cell: torch.Tensor,
                 pct_t: torch.Tensor, ret: Optional[torch.Tensor] = None,
                 win_lab: Optional[torch.Tensor] = None,
                 lab64: Optional[torch.Tensor] = None,
                 lab64_next: Optional[torch.Tensor] = None,
                 gate_mask: Optional[torch.Tensor] = None,
                 gate_valid: Optional[torch.Tensor] = None,
                 threat: Optional[torch.Tensor] = None,
                 expand: Optional[torch.Tensor] = None,
                 nums_next: Optional[torch.Tensor] = None,
                 w: Optional[torch.Tensor] = None,
                 lamb: dict = None) -> Tuple[torch.Tensor, dict]:
    """Weighted filtered-BC + value + win + gate + seg + dynamics + econ.

    IMPORTANT: `out` must come from forward(..., cell=cell, rtg=rtg_target,
    return_all=True) so pct_params sit at the recorded cell and every head
    is RTG-conditioned. `w` = AWR/survivor/kill weights from
    data_adapters.frame_weights. All aux targets derive from free shard
    labels (lab[t+1], episode outcomes) — see data_adapters.py.
    """
    lamb = lamb or dict(kind=0.5, cell=1.0, pct=0.25, val=0.1, win=0.1,
                        gate=0.5, seg=0.5, next_seg=0.4, threat=0.2,
                        expand=0.2, econ=0.05, ent=0.003)
    B = kind.shape[0]
    w = w if w is not None else torch.ones(B, device=kind.device)
    terms = {}

    kind_nll = F.cross_entropy(out["kind_logits"], kind, reduction="none",
                               label_smoothing=0.05)
    # 2026-08-14 pipeline-agent fix: cell_logits is (B,3,g*g); CE over dim=1
    # treated the 3 KINDS as classes and crashed (RuntimeError: Expected
    # target size [B, g*g]). Select the taken-kind row -> p(cell | kind).
    cell_logits_k = out["cell_logits"][torch.arange(B, device=kind.device), kind]
    cell_nll = F.cross_entropy(cell_logits_k, cell, reduction="none")
    pct_nll = -(Beta(out["pct_params"][0], out["pct_params"][1])
                .log_prob(pct_t.clamp(EPS, 1 - EPS)))
    pct_nll = pct_nll * (kind != 2).float()          # bank: pct undefined

    loss = (lamb["kind"] * kind_nll + lamb["cell"] * cell_nll
            + lamb["pct"] * pct_nll)
    terms["kind_nll"] = kind_nll.mean().detach()
    terms["cell_nll"] = cell_nll.mean().detach()
    terms["pct_nll"] = pct_nll.mean().detach()

    if ret is not None:
        val_loss = F.huber_loss(out["value"], ret, reduction="none")
        loss = loss + lamb["val"] * val_loss
        terms["val"] = val_loss.mean().detach()

    if win_lab is not None:
        win_nll = F.cross_entropy(out["win_logits"], win_lab,
                                  reduction="none", label_smoothing=0.05)
        loss = loss + lamb["win"] * win_nll
        terms["win"] = win_nll.mean().detach()

    if out["gate"] is not None and gate_mask is not None:
        g_bce = F.binary_cross_entropy(out["gate"], gate_mask,
                                       reduction="none").mean(dim=(2, 3))
        g_bce = g_bce * gate_valid.float()
        denom = gate_valid.float().sum() + EPS
        loss = loss + lamb["gate"] * (g_bce.sum() / denom).expand(B)
        terms["gate"] = (g_bce.sum() / denom).detach()

    if out["seg"] is not None and lab64 is not None:
        seg_ce = F.cross_entropy(out["seg"], lab64, reduction="none")
        # 2026-08-14 fix: CE with spatial target returns (B,H,W); reduce to
        # (B,) or it broadcasts against the (B,) loss into garbage.
        seg_ce = seg_ce.mean(dim=(1, 2))
        loss = loss + lamb["seg"] * seg_ce
        terms["seg"] = seg_ce.mean().detach()

    if out["next_seg"] is not None and lab64_next is not None:
        nseg_ce = F.cross_entropy(out["next_seg"], lab64_next,
                                  reduction="none").mean(dim=(1, 2))
        loss = loss + lamb["next_seg"] * nseg_ce
        terms["next_seg"] = nseg_ce.mean().detach()

    if out["threat"] is not None and threat is not None:
        th = F.binary_cross_entropy(out["threat"], threat,
                                    reduction="none").mean(dim=(1, 2))
        loss = loss + lamb["threat"] * th
        terms["threat"] = th.mean().detach()

    if out["expand"] is not None and expand is not None:
        ex = F.binary_cross_entropy(out["expand"], expand,
                                    reduction="none").mean(dim=(1, 2))
        loss = loss + lamb["expand"] * ex
        terms["expand"] = ex.mean().detach()

    if out["econ"] is not None and nums_next is not None:
        econ_l1 = F.smooth_l1_loss(out["econ"], nums_next, reduction="none")
        econ_l1 = econ_l1.mean(dim=1)
        loss = loss + lamb["econ"] * econ_l1
        terms["econ"] = econ_l1.mean().detach()

    def _cat_ent(p: torch.Tensor) -> torch.Tensor:
        # 2026-08-14 fix: F.categorical_entropy does not exist in shipping
        # torch builds -> AttributeError. Manual categorical entropy.
        return -(p * torch.log(p.clamp_min(EPS))).sum(dim=-1)

    marg = torch.logsumexp(out["cell_logits"], dim=1)
    ent = (_cat_ent(out["kind_probs"]) +
           _cat_ent(F.softmax(marg, dim=1)) +
           Beta(out["pct_params"][0], out["pct_params"][1]).entropy())
    loss = loss - lamb["ent"] * ent
    terms["entropy"] = ent.mean().detach()

    total = (loss * w).mean()
    terms["total"] = total.detach()
    return total, terms


# --------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------

def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def param_breakdown(net: Sovereign) -> dict:
    """Params per design module."""
    n = lambda m: sum(p.numel() for p in m.parameters())
    spine = n(net.stem) + sum(n(m) for m in net.enc)
    cortex = n(net.cortex) + n(net.film)
    decoder = sum(n(m) for m in net.dec)
    heads = (n(net.heat_mixer) + sum(n(m) for m in net.kind_towers) +
             n(net.kind_head) + n(net.value_head) + n(net.win_head) +
             n(net.pct_head))
    if net.cfg.use_gate:
        heads += n(net.gate_head)
    if net.cfg.use_aux_seg:
        heads += n(net.seg_head)
    if net.cfg.use_dynamics:
        heads += (n(net.next_seg_head) + n(net.threat_head) +
                  n(net.expand_head))
    if net.cfg.use_econ:
        heads += n(net.econ_head)
    if net.cfg.use_cell_value:
        heads += n(net.cell_value)
    mem = (n(net.mem) + n(net.mem_proj)) if net.mem is not None else 0
    planning = n(net.planning) if net.planning is not None else 0
    return dict(spine=spine, cortex=cortex, decoder=decoder, heads=heads,
                mem=mem, planning=planning)


def make_model(cfg: Optional[SovereignConfig] = None,
               seed: Optional[int] = None) -> Sovereign:
    if seed is not None:
        torch.manual_seed(seed)
    return Sovereign(cfg)


if __name__ == "__main__":
    # CPU smoke test: instantiate, count, run a fake frame through every path.
    fast = "--fast" in sys.argv
    cfg = SovereignConfig(map_size=128) if fast else SovereignConfig()
    m = make_model(cfg)
    total = count_params(m)
    bd = param_breakdown(m)
    print(f"params: {total/1e6:.2f}M (spine {bd['spine']/1e6:.1f}M | "
          f"cortex {bd['cortex']/1e6:.1f}M | decoder {bd['decoder']/1e6:.1f}M "
          f"| heads {bd['heads']/1e6:.1f}M | mem {bd['mem']/1e6:.1f}M | "
          f"planning {bd['planning']/1e6:.1f}M)")
    H = cfg.map_size
    rgb = torch.rand(2, 3, H, H)
    nums = torch.rand(2, 8)
    rtg = torch.randn(2)
    # compat path
    click, kind, pct, value = m(rgb, nums, rtg=rtg)
    print("compat out:", click.shape, kind.shape, pct.shape, value.shape)
    # full dict
    out = m(rgb, nums, rtg=rtg, cell=torch.tensor([120, 400]),
            return_all=True)
    print("full dict keys:", sorted(out.keys()))
    # act + memory
    a = m.act(rgb, nums, rtg=rtg)
    print("act:", {k: (v.shape if hasattr(v, "shape") else v)
                   for k, v in a.items() if k not in ("kind_probs",)})
    a2 = m.act(rgb, nums, rtg=rtg, state=a["state"])
    print("mem carry ok:", a2["state"].shape)
    # planning block smoke (encode -> predict -> dynamics round trip)
    z = m.encode(rgb, nums, rtg=rtg)
    prior_z, v_z, win_z = m.prediction(z)
    z2, r_hat = m.dynamics(z, torch.zeros(2, dtype=torch.long),
                           torch.zeros(2, dtype=torch.long),
                           torch.full((2,), 0.25))
    print("planning:", prior_z.shape, v_z.shape, win_z.shape, z2.shape,
          r_hat.shape)
    # no-rtg path (contract default)
    _, _, _, _ = m(rgb, nums)
    print("no-rtg contract path OK")
    if not fast:
        rgb256 = torch.rand(1, 3, 256, 256)
        _, _, _, _ = m(rgb256, torch.rand(1, 8), rtg=torch.randn(1))
        print("256x256 contract path OK")
    print("smoke OK")
