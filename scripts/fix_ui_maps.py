#!/usr/bin/env python3
"""2026-08-14: remove lobby-UI pollution baked into two real maps.

Investigation (do NOT re-litigate without evidence):
- mare_nostrum  : source screenshot is a LOBBY SCREEN (menu text + panels).
                  Not cleanable without losing half the frame -> gated out of
                  play via maps_meta pass=false. Detector template stays.
- caucasia      : left ~10% column holds menu text strokes; real map is
                  intact to the right. Clean: left band -> water.
- world2        : thin page-border ring around the frame. Clean: ring -> water.

Before cleaning, the POLLUTED land masks are snapshotted to
weights/maps/ui_tpl_<slug>.npy so audit_data.arena_eps_of can still flag OLD
shard episodes recorded on the polluted versions.

Reversible: original npz remain in git history.
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[1] / "weights" / "maps"


def save_polluted_tpl(slug: str):
    w = np.load(BASE / f"{slug}.npz", allow_pickle=True)["world"].astype(np.int16)
    land = (w >= 0).astype(np.uint8) * 255
    lr = np.array(Image.fromarray(land).resize((256, 256), Image.BILINEAR)) > 127
    np.save(BASE / f"ui_tpl_{slug}.npy", lr)
    print(f"snapshot polluted template {slug}: land256={int(lr.sum())}")


def clean(slug: str, fn):
    p = BASE / f"{slug}.npz"
    d = np.load(p, allow_pickle=True)
    w = d["world"].astype(np.int16)
    before = float((w >= 0).mean())
    fn(w)
    after = float((w >= 0).mean())
    np.savez(p, world=w.astype(np.int8))
    print(f"cleaned {slug}: land {before:.3f} -> {after:.3f}")


def main():
    # snapshot polluted masks first (templates for the detector)
    for slug in ("caucasia", "world2"):
        if not (BASE / f"ui_tpl_{slug}.npy").exists():
            save_polluted_tpl(slug)

    def caucasia_cut(w):
        # left menu-text band (measured on render: strokes in x < ~8%)
        w[:, : int(w.shape[1] * 0.10)] = -1

    def world2_ring(w):
        r = max(2, int(w.shape[0] * 0.02))
        w[:r, :] = -1
        w[-r:, :] = -1
        w[:, :r] = -1
        w[:, -r:] = -1

    clean("caucasia", caucasia_cut)
    clean("world2", world2_ring)

    # gate mare_nostrum out of the worker rotation
    meta_p = BASE / "maps_meta.json"
    meta = json.loads(meta_p.read_text())
    if "mare_nostrum" in meta:
        meta["mare_nostrum"]["pass"] = False
        meta["mare_nostrum"]["ui_polluted"] = True
        meta_p.write_text(json.dumps(meta, indent=1))
        print("mare_nostrum pass=false (lobby screenshot, ungrowable)")
    print("done; commit weights/maps changes")


if __name__ == "__main__":
    main()
