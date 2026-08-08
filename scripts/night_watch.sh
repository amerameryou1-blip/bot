#!/bin/bash
# Overnight loop-keeper: every 30 min, relaunch any dead RL kernels and
# ferry pending recordings (watchdog handles both). Runs ~12h.
cd "$(dirname "$0")/.." || exit 1
end=$(( $(date +%s) + 12 * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  echo "== watchdog $(date +%H:%M) ==" >> /tmp/night_watch.log
  python3 scripts/watchdog_loop.py --relaunch >> /tmp/night_watch.log 2>&1
  sleep 1800
done
echo "night_watch done" >> /tmp/night_watch.log
