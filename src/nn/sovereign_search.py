"""
search.py — MCTS planning for SOVEREIGN v3 (MuZero-lite).

At decision time the model can now THINK: it encodes the board into a
compressed latent, then runs PUCT tree search over the (kind, cell) action
space using the learned prediction/dynamics networks (see model.py's
`encode` / `prediction` / `dynamics`). This is AlphaZero-style planning
adapted to a 3x1024 discrete action space with a policy prior:

    prior      — from the real policy at the ROOT (legality already
                 folded in via the gate/mask), from the prediction net at
                 deeper nodes
    value      — win-probability head (the last-survivor objective itself)
    dynamics   — one-step latent transition trained on consecutive shard
                 frames (rgb_t, a_t -> rgb_{t+1}) plus the farmer reward

The search visits become policy-improvement targets for PPO (AlphaZero's
policy improvement operator, see train_notes.md Stage B).

Cost: 64 sims x depth 12 x (~27M-flop dynamics+prediction) ≈ 20-40ms on a
T4 in fp16 — well inside the 560ms tick budget. Honest simplifications
(documented, not hidden): (1) the rollout uses a fixed 25% troop commit
(the pct of the FINAL chosen action still comes from the full policy head
at that cell); (2) the latent is a vector (cortex pool + 8x8 compressed
map) — a MuZero-grade abstract state, not a pixel-perfect one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-6


@dataclass
class SearchResult:
    kind: int
    cell: int
    pct: float
    expected_win: float          # mean Q of the visit distribution
    q_chosen: float              # Q of the chosen action
    visits: torch.Tensor         # (3*g*g,) visit counts
    visit_probs: torch.Tensor    # (3*g*g,) normalized visit distribution
    max_depth_reached: int
    log: dict = field(default_factory=dict)


class _Node:
    __slots__ = ("z", "lp", "n", "w", "children", "depth")

    def __init__(self, z: torch.Tensor, lp: float, depth: int):
        self.z = z                # latent state (1, latent_dim)
        self.lp = lp              # log-prob of this edge (prior)
        self.n = 0                # visit count
        self.w = 0.0              # accumulated value
        self.children: Dict[int, _Node] = {}
        self.depth = depth

    @property
    def q(self) -> float:
        return self.w / max(self.n, 1)


class Searcher:
    def __init__(self, model, sims: int = 64, max_depth: int = 12,
                 c_puct: float = 1.25, top_k: int = 16,
                 dirichlet_alpha: float = 0.3, dirichlet_frac: float = 0.25,
                 discount: float = 0.995, temperature: float = 1.0,
                 default_pct: float = 0.25, seed: Optional[int] = None):
        self.model = model
        self.sims = sims
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.top_k = top_k
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.discount = discount
        self.temperature = temperature
        self.default_pct = default_pct
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        self.grid = model.cfg.grid_default
        self.num_actions = model.cfg.num_kinds * self.grid * self.grid

    # -- action packing -----------------------------------------------------

    def _kind_of(self, a: int) -> torch.Tensor:
        return torch.tensor([a // (self.grid * self.grid)],
                            dtype=torch.long, device=self._device())

    def _cell_of(self, a: int) -> torch.Tensor:
        return torch.tensor([a % (self.grid * self.grid)],
                            dtype=torch.long, device=self._device())

    def _device(self):
        return next(self.model.parameters()).device

    # -- main entry ---------------------------------------------------------

    @torch.no_grad()
    def search(self, rgb: torch.Tensor, nums: Optional[torch.Tensor] = None,
               rtg: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None,
               deterministic: bool = False) -> SearchResult:
        """Run MCTS from the current observation and return the chosen
        action + visit statistics. rgb must be a single sample (B=1)."""
        m = self.model
        assert rgb.shape[0] == 1, "search operates on a single sample (B=1)"
        device = rgb.device

        # full policy pass at the root: legality already inside cell_logits
        out = m(rgb, nums, rtg=rtg, mask=mask, grid=self.grid,
                return_all=True)
        root_logits = out["cell_logits"][0]              # (3*g*g,)
        z_root = m.encode(rgb, nums, rtg=rtg)            # (1, latent_dim)

        # root prior with Dirichlet exploration noise (global RNG: Dirichlet
        # .sample() takes no generator arg)
        lp_root = F.log_softmax(root_logits, dim=0)
        if self.dirichlet_alpha > 0:
            noise = torch.distributions.Dirichlet(
                torch.full((self.num_actions,), self.dirichlet_alpha,
                           device=device)).sample()
            lp_root = torch.log(
                (1.0 - self.dirichlet_frac) * torch.exp(lp_root) +
                self.dirichlet_frac * noise + EPS)

        root = _Node(z_root, 0.0, 0)   # lp unused at root
        root.lp = 0.0
        # store root log-probs for children priors
        root_p = lp_root

        max_depth_seen = 0
        for _ in range(self.sims):
            node = root
            path: List[Tuple[_Node, int]] = []
            depth = 0

            # ---------- select ----------
            while len(node.children) > 0 and depth < self.max_depth:
                best_a, best_u = None, -float("inf")
                tot_n = max(node.n, 1)
                for a, ch in node.children.items():
                    u = ch.q + self.c_puct * math.exp(ch.lp) * \
                        math.sqrt(tot_n) / (1.0 + ch.n)
                    if u > best_u:
                        best_u, best_a = u, a
                if best_a is None:
                    break
                node = node.children[best_a]
                path.append((node, best_a))
                depth += 1
            max_depth_seen = max(max_depth_seen, depth)

            # ---------- expand + evaluate ----------
            if depth < self.max_depth and node.depth < self.max_depth:
                if node is root:
                    prior_lp = root_p
                else:
                    prior_z, _, win_z = m.prediction(node.z)
                    prior_lp = F.log_softmax(prior_z[0], dim=0)
                    win_prob = torch.softmax(win_z, dim=1)[0, 1].item()
                    v = win_prob
                if node is not root:
                    pass
                else:
                    # root evaluation: policy win prob from the full model
                    v = out["win_prob"][0].item()
                # children from top-k prior
                k = min(self.top_k, self.num_actions)
                top = torch.topk(prior_lp, k)
                for a in top.indices.tolist():
                    a = int(a)
                    if a in node.children:
                        continue
                    z_next, _r = m.dynamics(
                        node.z, self._kind_of(a), self._cell_of(a),
                        torch.tensor([self.default_pct], device=device))
                    node.children[a] = _Node(z_next,
                                             float(prior_lp[a]),
                                             node.depth + 1)
                # if no child was legal/creatable, evaluate the node itself
                if not node.children:
                    _, _, win_z = m.prediction(node.z)
                    v = torch.softmax(win_z, dim=1)[0, 1].item()
            else:
                # depth limit: evaluate the leaf
                _, _, win_z = m.prediction(node.z)
                v = torch.softmax(win_z, dim=1)[0, 1].item()

            # ---------- backprop ----------
            for nd, _a in reversed(path):
                nd.n += 1
                nd.w += v
                v *= self.discount
            root.n += 1
            root.w += v

        # ---------- outcome ----------
        visits = torch.zeros(self.num_actions, device=device)
        for a, ch in root.children.items():
            visits[a] = ch.n
        if self.temperature <= 0 or deterministic:
            chosen = int(torch.argmax(visits).item())
        else:
            p = torch.pow(visits.clamp_min(0.0), 1.0 / self.temperature)
            s = p.sum()
            if s <= EPS:
                chosen = int(torch.argmax(root_p).item())
            else:
                chosen = int(torch.multinomial(
                    p / s, 1, generator=self.rng).item())
        probs = visits / max(visits.sum().item(), EPS)

        kind = chosen // (self.grid * self.grid)
        cell = chosen % (self.grid * self.grid)

        # final pct from the full policy head at the chosen cell
        H = m.cfg.map_size
        gd = self.grid
        y = (cell // gd) * (H // gd) + (H // gd) // 2
        xc = (cell % gd) * (H // gd) + (H // gd) // 2
        yx = torch.stack([torch.tensor([y]), torch.tensor([xc])],
                         dim=0).unsqueeze(0).to(device)
        pct_feat = m._gather(out["dec_feat"], yx)
        ctx_full = m._ctx_full(nums if nums is not None else
                               torch.zeros(1, m.cfg.ctx_dim, device=device),
                               rtg)
        kind_oh = F.one_hot(torch.tensor([kind], device=device),
                            m.cfg.num_kinds).float()
        alpha, beta_p = m.pct_head(pct_feat, ctx_full, kind_oh)
        pct = float((alpha / (alpha + beta_p))[0].item())

        qs = torch.zeros(self.num_actions, device=device)
        for a, ch in root.children.items():
            qs[a] = ch.q
        expected = float((qs * probs).sum().item())
        q_chosen = float(qs[chosen].item())
        ent = float(-(probs * torch.log(probs + EPS)).sum().item())

        return SearchResult(
            kind=kind, cell=cell, pct=pct,
            expected_win=expected, q_chosen=q_chosen,
            visits=visits, visit_probs=probs,
            max_depth_reached=max_depth_seen,
            log=dict(sims=self.sims, prior_entropy=ent,
                     mean_visits=visits.max().item()))


if __name__ == "__main__":
    # CPU smoke test with a tiny config (full 256px needs a GPU for speed)
    try:
        from model import Sovereign, SovereignConfig, make_model
    except ImportError:
        from nn.model import Sovereign, SovereignConfig, make_model  # noqa

    cfg = SovereignConfig(map_size=64)
    cfg.cortex_layers = 2
    cfg.enc_ch = (32, 32, 64, 96, 160, 224)
    cfg.enc_blocks = (1, 1, 1, 1, 1)
    cfg.dec_plan = ((192, 160, 1), (128, 96, 1), (96, 64, 1),
                    (64, 32, 1), (48, None, 1), (32, None, 1))
    cfg.dec_out_ch = 32
    cfg.head_hidden = 512
    cfg.cortex_d = 384
    cfg.cortex_ffn = 1024
    cfg.grid_default = 16
    cfg.use_mem = False
    m = make_model(cfg, seed=0)
    s = Searcher(m, sims=4, max_depth=3, top_k=4)
    rgb = torch.rand(1, 3, 64, 64)
    res = s.search(rgb, torch.rand(1, 8))
    print("search smoke:", res.kind, res.cell, round(res.pct, 3),
          "expected_win", round(res.expected_win, 3),
          "visits", res.visits.sum().item(), "depth", res.max_depth_reached)
    print("search.py smoke OK")
