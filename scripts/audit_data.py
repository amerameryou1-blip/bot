#!/usr/bin/env python3
"""FULL DATA AUDIT (user order 2026-08-09): verify recordings + shards
BEFORE training. Eyes + numbers. Prints PASS/FAIL per check.

Usage:
  python3 scripts/audit_data.py --recordings DIR            # audit local folder
  python3 scripts/audit_data.py --recordings DIR --purge    # also delete FAIL
                                                            # sessions from HF
  python3 scripts/audit_data.py --shards 5                  # audit N newest
                                                            # rl shards from HF
A click is UI-GARBAGE if inside: top banner (y<50), leaderboard (x<300,y<320),
bottom bar (y>740) — those opened modals / did nothing (match #6 post-mortem).
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

CROWN = (240, 224, 112)


def ui_zone(x, y):
    return y < 50 or (x < 300 and y < 320) or y > 740


def audit_session(sess: Path):
    fails, warns = [], []
    meta_p = sess / "meta.json"
    if not meta_p.exists():
        return ["no meta.json"], []
    meta = json.load(open(meta_p))
    sc = meta.get("self_color") or [0, 0, 0]
    if not meta.get("camera_pass"):
        fails.append("camera gate failed")
    if sum(abs(a - b) for a, b in zip(sc, CROWN)) < 90:
        fails.append("self_color is the CROWN icon (pre-fix bug)")
    if sum(abs(a - b) for a, b in zip(sc, (48, 180, 24))) < 30:
        fails.append("self_color untrusted (row-highlight green, pre-fix)")
    frames = sorted((sess / "frames").glob("*.jpg"))
    if not frames:
        fails.append("no frames")
        return fails, warns
    # self visibility over sampled frames
    covs = []
    lite = [int(v + (255 - v) * 0.55) for v in sc]
    for f in frames[:: max(1, len(frames) // 15)]:
        img = np.asarray(Image.open(f).convert("RGB")).astype(int)
        m = (np.abs(img - np.array(sc)).max(axis=2) < 24) | \
            (np.abs(img - np.array(lite)).max(axis=2) < 28)
        covs.append(float(m.mean()))
    med = float(np.median(covs))
    if med < 0.004:
        fails.append(f"bot invisible (median self coverage {med:.4f})")
    elif med < 0.01:
        warns.append(f"bot barely visible ({med:.4f})")
    # click validity
    bad = tot = 0
    for line in (sess / "clicks.jsonl").read_text().strip().split("\n"):
        if not line.strip():
            continue
        c = json.loads(line)
        tot += 1
        if ui_zone(c["x"], c["y"]) or not (0 <= c["x"] < 1280 and 0 <= c["y"] < 800):
            bad += 1
    if tot == 0:
        warns.append("no clicks")
    elif bad / tot > 0.2:
        fails.append(f"{bad}/{tot} clicks in UI zones (modal garbage)")
    elif bad:
        warns.append(f"{bad}/{tot} clicks in UI zones (filtered at label time)")
    return fails, warns


def audit_shard(path):
    d = np.load(path)
    probs = []
    rgb = d["rgb"]
    if rgb.dtype == np.uint8:
        pass  # uint8 by design — trainer/unpack divide by 255
    elif rgb.dtype == np.float32 and rgb.max() <= 1.0 + 1e-3 and rgb.min() >= 0:
        pass
    else:
        probs.append(f"rgb range bad ({rgb.min():.2f}..{rgb.max():.2f})")
    if int(d["lens"].sum()) != len(rgb):
        probs.append("lens sum != rgb len")
    if not np.isin(d["kind"], [0, 1, 2]).all():
        probs.append("kind out of range")
    if (d["cell"].min() < 0) or (d["cell"].max() > 255):
        probs.append("cell out of range")
    if (d["pct"].min() < 0) or (d["pct"].max() > 1):
        probs.append("pct out of range")
    if not (np.isfinite(d["reward"]).all() and np.isfinite(d["logp"]).all()):
        probs.append("non-finite reward/logp")
    return probs


def audit_v2_shard(path):
    """Size-agnostic (128/192/256 px, grid=size//8, REC_EVERY 1-4)."""
    d = np.load(path)
    probs = []
    size = d["rgb"].shape[1]
    grid = size // 8
    if d["rgb"].dtype != np.uint8:
        probs.append("rgb not uint8")
    if size not in (128, 192, 256) or d["rgb"].shape[2] != size or d["rgb"].shape[3] != 3:
        probs.append(f"rgb shape {d['rgb'].shape}")
    if d["lab"].shape[1:] != (size, size):
        probs.append("lab shape mismatch")
    if not np.isin(d["lab"], [0, 1, 2, 3]).all():
        probs.append("lab out of range")
    if not np.isin(d["kind"], [0, 1, 2]).all():
        probs.append("kind out of range")
    if (d["cell"].min() < 0) or (d["cell"].max() >= grid * grid):
        probs.append("cell out of range")
    ls, rl = int(d["lens"].sum()), len(d["rgb"])
    if not (rl <= ls <= rl * 4):
        probs.append(f"lens/rgb ratio odd ({ls} vs {rl})")
    if not (np.isfinite(d["reward"]).all() and np.isfinite(d["nums"]).all()):
        probs.append("non-finite reward/nums")
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings")
    ap.add_argument("--shards", type=int, default=0)
    ap.add_argument("--v2", type=int, default=0, help="audit N newest v2 shards")
    ap.add_argument("--purge", action="store_true")
    a = ap.parse_args()

    if a.v2:
        tok = os.environ.get("HF_TOKEN", "").strip()
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=tok)
        fs = sorted(f for f in api.list_repo_files(
            "amer224/territorial-bot-data", repo_type="dataset", token=tok)
            if f.startswith("rl/shards_v2/"))[-a.v2:]
        bad = 0
        for f in fs:
            p = hf_hub_download("amer224/territorial-bot-data", f,
                                repo_type="dataset", token=tok)
            probs = audit_v2_shard(p)
            bad += bool(probs)
            print(f"[{'FAIL' if probs else 'PASS'}] {f}: {'; '.join(probs) or 'clean'}")
        print("V2 AUDIT DONE", "— CLEAN" if not bad else f"— {bad} BAD")
        return

    PURGE_MARKS = ("camera gate failed", "CROWN", "invisible", "untrusted")
    bad_sessions, purge_list = [], []
    if a.recordings:
        for sess in sorted(Path(a.recordings).glob("*/meta.json")):
            s = sess.parent
            fails, warns = audit_session(s)
            tag = "FAIL" if fails else ("WARN" if warns else "PASS")
            print(f"[{tag}] {s.name}: {'; '.join(fails + warns) or 'clean'}")
            if fails:
                bad_sessions.append(s.name)
                if any(m in ";".join(fails) for m in PURGE_MARKS):
                    purge_list.append(s.name)
    if a.shards:
        tok = os.environ.get("HF_TOKEN", "").strip()
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=tok)
        fs = sorted(f for f in api.list_repo_files(
            "amer224/territorial-bot-data", repo_type="dataset", token=tok)
            if f.startswith("rl/shards/"))[-a.shards:]
        for f in fs:
            p = hf_hub_download("amer224/territorial-bot-data", f,
                                repo_type="dataset", token=tok)
            probs = audit_shard(p)
            print(f"[{'FAIL' if probs else 'PASS'}] {f}: {'; '.join(probs) or 'clean'}")
    # purge only unrecoverable sessions (invisible bot / wrong self color /
    # failed camera). Garbage-CLICK sessions stay: frames still train vision,
    # bad clicks are filtered at label time.
    if a.purge and purge_list:
        bad_sessions = purge_list
        tok = os.environ.get("HF_TOKEN", "").strip()
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        allf = api.list_repo_files("amer224/territorial-bot-data",
                                   repo_type="dataset", token=tok)
        for sid in bad_sessions:
            doomed = [f for f in allf if f.startswith(f"recordings/{sid}/")]
            if not doomed:
                continue
            try:
                api.delete_folder(path_in_repo=f"recordings/{sid}",
                                  repo_id="amer224/territorial-bot-data",
                                  repo_type="dataset", token=tok,
                                  commit_message=f"audit purge {sid}")
            except Exception:
                for p in doomed:
                    try:
                        api.delete_file(path_in_repo=p,
                                        repo_id="amer224/territorial-bot-data",
                                        repo_type="dataset", token=tok)
                    except Exception as e2:
                        print(f"  del fail {p}: {str(e2)[:60]}")
            print(f"purged {sid} ({len(doomed)} files)")
    print("AUDIT DONE")


if __name__ == "__main__":
    main()


def _ui_templates():
    """2026-08-14: lobby-screenshot maps that slipped the stat gate.
    mare_nostrum's source png is a LOBBY SCREEN (menu text/panels baked in);
    caucasia had a left menu sliver; world2 a page-border ring (both since
    cleaned by scripts/fix_ui_maps.py — templates kept so OLD episodes on the
    polluted masks still get flagged). Returns {slug: land_mask_256}."""
    from PIL import Image
    import os
    global _UI_TPL
    if _UI_TPL is None:
        _UI_TPL = {}
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "weights", "maps")
        # ONLY ruined lobby-screen maps here (gated out of play forever, so
        # no over-filter risk): black/white_arena + mare_nostrum.
        # Sliver maps (caucasia/world2/world/mountains) are surgically
        # cleaned but STAY in rotation; template-flagging them would also
        # flag their new clean episodes (water masks ~identical) and kill
        # the map. Old sliver episodes keep ~1% cosmetic label noise.
        for slug in ("black_arena", "white_arena", "mare_nostrum"):
            p = os.path.join(base, slug + ".npz")
            if not os.path.exists(p):
                continue
            w = np.load(p, allow_pickle=True)["world"].astype(np.int16)
            water = (w < 0).astype(np.uint8) * 255
            _UI_TPL[slug] = (np.array(Image.fromarray(water).resize(
                (256, 256), Image.BILINEAR)) > 127)
    return _UI_TPL


_UI_TPL = None


_GOOD_TPL = None


def _good_templates():
    """2026-08-15: water masks of the 21 maps the agent rendered and
    eyeballed as CLEAN gameplay (every pass's samples). Used to let the
    unattended daemon auto-commit ONLY shards whose episodes all
    fingerprint-match a reviewed map (IoU>0.90) with perfect stats."""
    global _GOOD_TPL
    if _GOOD_TPL is None:
        import os
        from PIL import Image
        _GOOD_TPL = {}
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "weights", "maps")
        try:
            meta = json.load(open(os.path.join(base, "maps_meta.json")))
        except Exception:
            return _GOOD_TPL
        for slug, v in meta.items():
            if not v.get("pass"):
                continue
            p = os.path.join(base, slug + ".npz")
            if not os.path.exists(p):
                continue
            w = np.load(p, allow_pickle=True)["world"].astype(np.int16)
            water = (w < 0).astype(np.uint8) * 255
            _GOOD_TPL[slug] = (np.array(Image.fromarray(water).resize(
                (256, 256), Image.BILINEAR)) > 127)
    return _GOOD_TPL


def good_map_eps_of(lab, lens):
    """Per-episode True if the episode's water mask matches a reviewed
    CLEAN map (IoU > 0.90). Water is invariant mid-episode, so this is a
    stable fingerprint."""
    rec = [max(1, int(l) // 2) for l in lens]
    out = []
    off = 0
    tpls = _good_templates()
    for e in range(len(lens)):
        n = rec[e]
        mid = off + n // 2
        ok = False
        if mid < lab.shape[0] and tpls:
            wf = lab[mid] == 0
            for wm in tpls.values():
                inter = (wf & wm).sum()
                union = (wf | wm).sum()
                if union > 0 and inter / union > 0.90:
                    ok = True
                    break
        out.append(bool(ok))
        off += n
    return out


def _tpl_flags(lab_mid):
    """True if the frame's WATER mask matches a polluted-UI map template.
    Water never changes during an episode (players only eat land), so the
    water mask is a stable per-map fingerprint — land masks drift as blobs
    grow and both miss polluted eps and false-flag clean ones."""
    wf = lab_mid == 0
    for slug, wm in _ui_templates().items():
        inter = (wf & wm).sum()
        union = (wf | wm).sum()
        if union > 0 and inter / union > 0.90:
            return True
    return False


def arena_eps_of(lab, lens):
    """Per-episode True if that episode played on a lobby-screenshot map.
    Signatures measured on source maps: black_arena water .92 + land-thin .48
    (watermark text strokes); white_arena water .004; real watery maps
    (island_kingdom) land-thin .12.
    2026-08-14: + template match for mare_nostrum/caucasia/world2 lobby-UI
    pollution (found by eyeballing shard_v2_1786711002)."""
    rec = [max(1, int(l) // 2) for l in lens]
    out = []
    off = 0
    for e in range(len(lens)):
        n = rec[e]
        mid = off + n // 2
        a = False
        if mid < lab.shape[0]:
            m = lab[mid]
            wm = float((m == 0).mean())
            lm = (m == 1)
            core = lm[1:-1, 1:-1]
            er = (core & lm[:-2, 1:-1] & lm[2:, 1:-1]
                  & lm[1:-1, :-2] & lm[1:-1, 2:])
            thin = 1 - er.sum() / max(lm.sum(), 1)
            a = wm < 0.05 or (wm > 0.88 and thin > 0.35) or _tpl_flags(m)
        out.append(bool(a))
        off += n
    return out
