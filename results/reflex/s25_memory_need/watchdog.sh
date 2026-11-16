#!/bin/bash
# Kills the S25 workers if the shared DGX runs low on memory (MemAvailable < 30 GB); logs to watchdog.log.
R=${REPO:-.}/results/reflex/s25_memory_need
while pgrep -f "s25_(memory|generic|recal).py" > /dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "${a:-0}" -lt 30 ]; then echo "$(date -u) MemAvailable ${a} GB: killing S25 workers" >> $R/watchdog.log; pkill -f "s25_(memory|generic|recal).py"; fi
  sleep 5
done
echo "$(date -u) no S25 workers left; watchdog exits" >> $R/watchdog.log
