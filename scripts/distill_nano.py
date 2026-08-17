"""
distill_nano.py — wake-sleep distillation INTO SOVEREIGN-nano.

ALIGNED WITH THE PIPELINE'S FIXED VERSION (B, 15 Aug). The old code had
two real bugs, both caught by B's execution review:

  1. WRONG NORMALIZATION: the KL/CE terms softmaxed over the KIND dim of
     (B,3,g²) — cell_logits are kind-conditioned logits that must be
     normalized over the JOINT (kind × cell) space. Fixed: flatten to
     (B, 3*g²) and softmax over that.
  2. HARD-CELL CE CRASH (v3 review bug #2 again): CE((B,3,g²),(B,))
     crashes — the taken-kind row must be selected. `kind` is now a
     REQUIRED key in every batch for ce_cell.

The teacher (v3 290M or any contract-compatible net) emits everything via
forward(..., return_all=True): cell_logits (B,3,g²), kind_logits,
win_logits, value, pct_params, econ. Nano's contract is identical, so
every term is a direct op. search_targets (B,3,g²) are the teacher's MCTS
visit distributions when available (AlphaZero-lite policy improvement
flowing into the cheap policy).

torch only. The smoke uses two nano-sized nets and SHAPE-ASSERTS the
joint flattening before running the loss.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn.functional as F


def _joint(x: torch.Tensor) -> torch.Tensor:
    """(B,3,g*g) -> (B,3g*g) joint (kind × cell) logit vector."""
    return x.flatten(1)


def distill_nano(teacher, student, batches: Iterator[dict], epochs: int = 1,
                 lr: float = 3e-4, kl_cell: float = 1.0,
                 kl_kind: float = 1.0, kl_win: float = 0.5,
                 mse_val: float = 1.0, mse_pct: float = 0.5,
                 mse_econ: float = 0.1, ce_cell: float = 0.25,
                 search_w: float = 0.5, device: str = "cuda") -> None:
    """Joint-space distillation. Batches need: rgb, nums, kind, cell;
    optional rtg, search_targets."""
    teacher.to(device).eval()
    student.to(device).train()
    dev = torch.device(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)

    for ep in range(epochs):
        tot, nb = 0.0, 0
        for raw in batches:
            b = {k: v.to(dev) for k, v in raw.items()
                 if hasattr(v, "to") and k != "search_targets"}
            rgb, nums = b["rgb"], b["nums"]
            rtg = b.get("rtg")
            kind = b["kind"]                       # REQUIRED
            cell = b["cell"]

            with torch.no_grad():
                to = teacher(rgb, nums, rtg=rtg, cell=cell,
                             return_all=True)
            so = student(rgb, nums, rtg=rtg, cell=cell, return_all=True)

            t_joint = _joint(to["cell_logits"])     # (B, 3g²)
            s_joint = _joint(so["cell_logits"])

            loss = kl_cell * F.kl_div(
                F.log_softmax(s_joint, 1), F.softmax(t_joint, 1),
                reduction="batchmean")
            loss = loss + kl_kind * F.kl_div(
                F.log_softmax(so["kind_logits"], 1),
                F.softmax(to["kind_logits"], 1), reduction="batchmean")
            loss = loss + kl_win * F.kl_div(
                F.log_softmax(so["win_logits"], 1),
                F.softmax(to["win_logits"], 1), reduction="batchmean")
            loss = loss + mse_val * F.mse_loss(so["value"], to["value"])
            loss = loss + mse_pct * (
                F.mse_loss(so["pct_params"][0], to["pct_params"][0]) +
                F.mse_loss(so["pct_params"][1], to["pct_params"][1]))
            if so["econ"] is not None and to["econ"] is not None:
                loss = loss + mse_econ * F.mse_loss(so["econ"], to["econ"])

            # hard-cell CE on the taken-kind row (bug 2 fix)
            loss = loss + ce_cell * F.cross_entropy(
                so["cell_logits"][torch.arange(len(kind), device=dev),
                                  kind.long()],
                cell.long())

            if "search_targets" in raw:
                st = _joint(raw["search_targets"].to(dev)).clamp_min(1e-6)
                loss = loss + search_w * F.kl_div(
                    F.log_softmax(s_joint, 1), st, reduction="batchmean")

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"distill_nano ep {ep+1}: loss={tot/max(nb,1):.4f}",
              flush=True)


def save_student(student, path: str) -> None:
    torch.save({"config": student.cfg, "state_dict": student.state_dict()},
               path)
    print(f"[distill_nano] saved {path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    # smoke: teacher = nano-width net, student = thinner nano — the real
    # v3 teacher swaps in without code changes (same contract). The joint
    # flattening is shape-asserted BEFORE the loss runs.
    from nn.sovereign_nano import NanoConfig, make_nano

    tcfg = NanoConfig(map_size=64)
    tcfg.grid_default = 16           # matches search_targets below
    scfg = NanoConfig(map_size=64)
    scfg.grid_default = 16           # contract grid for the smoke
    scfg.enc_ch = (12, 24, 32, 64, 96)
    scfg.dec_plan = ((64, 64, 1), (32, 32, 1), (24, 24, 1),
                     (20, None, 1), (20, None, 0))
    scfg.dec_out_ch = 20
    scfg.head_hidden = 96
    teacher = make_nano(tcfg, seed=0)
    student = make_nano(scfg, seed=1)
    batch = [dict(rgb=torch.rand(2, 3, 64, 64), nums=torch.rand(2, 8),
                  rtg=torch.rand(2, 1), kind=torch.tensor([0, 2]),
                  cell=torch.tensor([10, 40]),
                  search_targets=torch.rand(2, 3 * 16 * 16))]
    # shape asserts (pipeline's version does the same before KL)
    with torch.no_grad():
        probe = student(batch[0]["rgb"], batch[0]["nums"],
                        rtg=batch[0]["rtg"], return_all=True)
        assert probe["cell_logits"].shape == (2, 3, 256), \
            probe["cell_logits"].shape
        assert batch[0]["search_targets"].shape == (2, 3, 256), \
            batch[0]["search_targets"].shape
        assert _joint(probe["cell_logits"]).shape == (2, 768)
    distill_nano(teacher, student, iter(batch), epochs=1, device="cpu")
    print("distill_nano.py smoke OK (joint-flatten, kind-row CE)")
