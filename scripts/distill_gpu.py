#!/usr/bin/env python3
"""distill_gpu.py — wake-sleep distillation: winner 290M teacher -> nano.
A's §10 recipe verbatim: teacher=src/nn/sovereign.py from winner ckpt;
batches=sovereign_data.ShardPrepper (kind required); rtg=None both nets;
mse_pct/mse_econ=0 if teacher return_all lacks them. Uploads student ckpt.
Env: DISTILL_CKPT (HF path e.g. rl/sovereign/ckpt_final_X.pt),
     OUT_NAME (e.g. nano_v10_distill.pt), SH (shards dir).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from nn.sovereign import Sovereign
from nn.sovereign_nano import make_nano
from sovereign_data import ShardPrepper
from distill_nano import distill_nano


def batches(sh_dir, cap_shards=300):
    import glob
    for p in sorted(glob.glob(str(sh_dir / "*.npz")))[:cap_shards]:
        try:
            pre = ShardPrepper(p).prepare()
        except Exception as e:
            print("skip", Path(p).name, str(e)[:60], flush=True)
            continue
        if pre is None or len(pre.rgb) < 8:
            continue
        t = torch.from_numpy
        yield dict(rgb=t(pre.rgb), nums=t(pre.nums), cell=t(pre.cell),
                   kind=t(pre.kind))


def main():
    tok = os.environ.get("HF_TOKEN", "")
    ckpt_hf = os.environ.get("DISTILL_CKPT", "rl/sovereign/ckpt_final.pt")
    out_name = os.environ.get("OUT_NAME", "nano_distill.pt")
    sh = Path(os.environ.get("SH", "weights/nn/rl/sov_shards"))

    from huggingface_hub import hf_hub_download, upload_file
    ck = hf_hub_download("amer224/territorial-bot-data", ckpt_hf,
                         repo_type="dataset", token=tok,
                         local_dir=str(ROOT / "weights"))
    teacher = Sovereign()
    teacher.load_state_dict(torch.load(ck, map_location="cpu"))
    student = make_nano()
    distill_nano(teacher, student, batches(sh), epochs=1, device="cuda")
    out = Path("/tmp") / out_name
    torch.save({"config": student.cfg, "state_dict": student.state_dict()}, out)
    upload_file(path_or_fileobj=str(out), path_in_repo=f"rl/{out_name}",
                repo_id="amer224/territorial-bot-data", repo_type="dataset",
                token=tok)
    print("DISTILL DONE ->", f"rl/{out_name}", flush=True)


if __name__ == "__main__":
    main()
