#!/usr/bin/env python3
"""Export the trained model to Hugging Face Hub.

Pushes model.safetensors + config.json + a model card to
  amer224/territorial-bot-nn   (private)

Token: read from env HF_TOKEN (NOT stored in any repo file).
Run:  HF_TOKEN=hf_... python3 scripts/export_hf.py
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo, upload_file

WEIGHTS = REPO / "weights" / "nn"
MODEL_PT = WEIGHTS / "model.pt"
CONFIG = WEIGHTS / "config.json"
HF_REPO = "amer224/territorial-bot-nn"
TOKEN = os.environ.get("HF_TOKEN", "").strip()


def main():
    if not TOKEN:
        print("HF_TOKEN env var required")
        sys.exit(1)
    if not MODEL_PT.exists():
        print("no trained model yet:", MODEL_PT)
        sys.exit(1)

    # save safetensors
    state = torch.load(MODEL_PT, map_location="cpu")
    st_path = WEIGHTS / "model.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(st_path))
    print(f"saved safetensors: {st_path.stat().st_size/1e6:.2f} MB")

    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {"grid": 16, "context_dim": 3, "classes": 5}
    cfg_path = WEIGHTS / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    api = HfApi(token=TOKEN)
    try:
        create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True, token=TOKEN)
        print("repo ready:", HF_REPO)
    except Exception as e:
        print("create_repo:", e)

    for f in (st_path, cfg_path):
        upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                    repo_id=HF_REPO, repo_type="model", token=TOKEN)
        print("uploaded", f.name)

    # model card
    card = """---
license: mit
tags: [reinforcement-learning, game-ai, computer-vision]
---

# Territorial.io Bot — Neural Network

Small CNN (~85k params) that reads the game screen (64x64 RGB) and outputs:
- per-pixel segmentation (water / neutral / me / enemy / UI)
- self-localization (my position on the map)
- an action policy (which cell to click, expand/attack/bank, attack %)
- a value estimate

Trained in a headless simulator with the attack meta (see ATTACK_META.md):
behavior cloning of a strong heuristic teacher + PPO fine-tuning for
last-survivor wins.

Weights: `model.safetensors` · Config: `config.json`
"""
    card_path = WEIGHTS / "README.md"
    card_path.write_text(card)
    upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                repo_id=HF_REPO, repo_type="model", token=TOKEN)
    print("uploaded README.md")
    print("DONE -> https://huggingface.co/" + HF_REPO)


if __name__ == "__main__":
    main()
