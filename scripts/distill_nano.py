"""
distill_nano.py — wake-sleep distillation INTO SOVEREIGN-nano.

The v3 teacher (or any contract-compatible model: forward(rgb, nums,
rtg=None, cell=None, return_all=True) -> dict with cell_logits/
kind_logits/value/win_logits/econ/pct_params) distills into nano via:

    KL(cell_logits) + KL(kind) + KL(win) + MSE(value) + MSE(pct_params)
    + MSE(econ) + hard-cell CE + optional search-target KL

Nano's contract is identical to the teacher's, so every term is a direct
op — no adapters, no shape translation. search_targets (B,3,g*g) are the
teacher's MCTS visit distributions when available (AlphaZero-lite policy
improvement flowing into the cheap policy).

torch only. The smoke uses two nano-sized nets (no 290M import needed).
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn.functional as F


def distill_nano(teacher, student, batches: Iterator[dict], epochs: int = 1,
                 lr: float = 3e-4, kl_cell: float = 1.0,
                 kl_kind: float = 1.0, kl_win: float = 0.5,
                 mse_val: float = 1.0, mse_pct: float = 0.5,
                 mse_econ: float = 0.1, ce_cell: float = 0.25,
                 search_w: float = 0.5, device: str = "cuda") -> None:
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
            cell = b["cell"]
            with torch.no_grad():
                to = teacher(rgb, nums, rtg=rtg, cell=cell,
                             return_all=True)
            so = student(rgb, nums, rtg=rtg, cell=cell, return_all=True)

            # 2026-08-15 fix (B): cell_logits is (B,3,g*g) — the distill must
            # match the JOINT (kind x cell) distribution, so flatten and
            # softmax over the joint dim; the old kind-dim softmax was wrong.
            sj = so["cell_logits"].flatten(1)
            tj = to["cell_logits"].flatten(1)
            loss = kl_cell * F.kl_div(
                F.log_softmax(sj, 1),
                F.softmax(tj, 1), reduction="batchmean")
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
            if "kind" in b:
                g2 = so["cell_logits"].shape[-1]
                joint_t = (b["kind"] * g2 + cell).long()
                loss = loss + ce_cell * F.cross_entropy(sj, joint_t)
            if "search_targets" in raw:
                loss = loss + search_w * F.kl_div(
                    F.log_softmax(sj, 1),
                    raw["search_targets"].to(dev).clamp_min(1e-6),
                    reduction="batchmean")

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
    # smoke: teacher = nano-width net, student = thinner nano — the real
    # teacher swaps in without code changes (same contract).
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from nn.sovereign_nano import NanoConfig, make_nano

    tcfg = NanoConfig(map_size=64)
    scfg = NanoConfig(map_size=64)
    # 2026-08-15 (A's fix, relayed): pin grid so search_targets (B,3,g*g)
    # matches cell_logits; smoke shape-asserts BEFORE the KL term runs.
    tcfg.grid_default = 16
    scfg.grid_default = 16
    scfg.enc_ch = (12, 24, 32, 64, 96)
    scfg.dec_plan = ((64, 64, 1), (32, 32, 1), (24, 24, 1),
                     (20, None, 1), (20, None, 0))
    scfg.dec_out_ch = 20
    scfg.head_hidden = 96
    teacher = make_nano(tcfg, seed=0)
    student = make_nano(scfg, seed=1)
    st = torch.rand(2, 3 * 16 * 16)
    with torch.no_grad():
        probe = teacher(torch.rand(2, 3, 64, 64), torch.rand(2, 8),
                        rtg=torch.rand(2, 1), cell=torch.tensor([10, 40]),
                        return_all=True)
    assert probe["cell_logits"].shape == (2, 3, 256), \
        f"grid mismatch: {tuple(probe['cell_logits'].shape)} vs targets (2,3,256)"
    assert st.shape == (2, 3 * 256), st.shape
    batch = [dict(rgb=torch.rand(2, 3, 64, 64), nums=torch.rand(2, 8),
                  rtg=torch.rand(2, 1), cell=torch.tensor([10, 40]),
                  kind=torch.tensor([0, 1]), search_targets=st)]
    distill_nano(teacher, student, iter(batch), epochs=1, device="cpu")
    print("distill_nano.py smoke OK")
