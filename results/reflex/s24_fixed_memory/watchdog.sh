#!/bin/bash
# Kills the S24 workers if the shared DGX runs low on memory (MemAvailable < 16 GB); logs to watchdog.log.
R=${REPO:-.}/results/reflex/s24_fixed_memory
while pgrep -f "s24_(fixed_memory|stack|timing).py" > /dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "${a:-0}" -lt 16 ]; then echo "$(date -u) MemAvailable ${a} GB: killing S24 workers" >> $R/watchdog.log; pkill -f "s24_(fixed_memory|stack|timing).py"; fi
  sleep 5
done
echo "$(date -u) no S24 workers left; watchdog exits" >> $R/watchdog.log
