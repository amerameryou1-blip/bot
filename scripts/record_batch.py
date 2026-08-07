#!/usr/bin/env python3
"""Sequential real-match recorder — keeps the data stream flowing.

Each match: fresh headless browser -> enter custom scenario -> fix camera
(ZOOM_LEVEL alternates for scale variety) -> play N minutes -> record frames
+ clicks + meta. Crash-safe: one failed match never stops the batch.

Usage:
    python3 scripts/record_batch.py --games 4 --minutes 2 --tag island1
"""
import argparse, glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT = os.path.join(ROOT, "run_bot.py")
REC_DIR = os.path.join(ROOT, "recordings")


def session_summary() -> list:
    out = []
    for meta in sorted(glob.glob(os.path.join(REC_DIR, "*/meta.json"))):
        m = json.load(open(meta))
        out.append({
            "session": m.get("session_id"),
            "frames": m.get("frames", 0),
            "clicks": m.get("clicks", 0),
            "map": m.get("map"),
            "zoom": m.get("zoom_level", "?"),
            "cam_pass": m.get("camera_pass"),
            "self": m.get("self_color"),
            "enemies": len(m.get("enemy_colors", [])),
            "last_survivor": m.get("last_survivor", False),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--tag", type=str, default="batch")
    ap.add_argument("--zoom", type=str, default="alt",
                    help="alt (alternate 0/1), auto, 0, 1, 2")
    args = ap.parse_args()

    os.makedirs(REC_DIR, exist_ok=True)
    log = f"[record_batch {args.tag}] {args.games} games x {args.minutes} min"
    print(log, flush=True)

    ok, fail = 0, 0
    for g in range(1, args.games + 1):
        start = time.time()
        if args.zoom == "alt":
            zl = "1" if (g % 2 == 1) else "0"
        else:
            zl = args.zoom
        env = dict(os.environ, ZOOM_LEVEL=zl, PLAY_MINUTES=str(args.minutes),
                   BOT_NAME=f"AureliaBot{g}")
        print(f"\n===== MATCH {g}/{args.games} (zoom={zl}) =====", flush=True)
        try:
            r = subprocess.run([sys.executable, BOT, "--record", "--games", "1"],
                               cwd=ROOT, env=env, timeout=int(args.minutes * 60 + 420),
                               capture_output=True, text=True)
            lines = r.stdout.splitlines()
            cam = [l for l in lines if "[camera] FINAL" in l]
            spawn = [l for l in lines if "SPAWN DETECTED" in l]
            print("\n".join(cam[-2:]), flush=True)
            print("\n".join(spawn[-1:]), flush=True)
            if r.returncode != 0:
                print(f"MATCH {g}: subprocess rc={r.returncode} (stderr tail):",
                      r.stderr[-500:], flush=True)
                fail += 1
            else:
                ok += 1
        except subprocess.TimeoutExpired:
            print(f"MATCH {g}: TIMEOUT", flush=True)
            fail += 1
        except Exception as e:
            print(f"MATCH {g}: ERROR {e}", flush=True)
            fail += 1
        dur = time.time() - start
        print(f"MATCH {g} took {dur:.0f}s", flush=True)

    print("\n===== BATCH SUMMARY =====", flush=True)
    print(f"matches ok={ok} fail={fail}", flush=True)
    for s in session_summary():
        print(f"  {s['session']}: frames={s['frames']} clicks={s['clicks']} "
              f"map={s['map']} zoom={s['zoom']} cam_pass={s['cam_pass']} "
              f"enemies={s['enemies']} last_survivor={s['last_survivor']}", flush=True)
    print("BATCH DONE", flush=True)
    sys.exit(0 if ok > 0 else 1)


if __name__ == "__main__":
    main()
