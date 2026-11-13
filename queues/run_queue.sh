#!/bin/bash
# Run each line of a queue file as: python run_experiments.py <line>
# Lines whose output already exists are skipped.
# Usage: bash queues/run_queue.sh queues/queue_a.txt
cd "$(dirname "$0")/.."
while read -r line; do
  [ -z "$line" ] && continue
  tag=$(echo "$line" | sed -E 's/.*--tag ([^ ]+).*/\1/')
  cmd=$(echo "$line" | cut -d' ' -f1)
  case "$cmd" in
    train)   out="results/$tag.pkl" ;;
    exploit) out="results/br/$tag.json" ;;
    analyse) out="results/${tag}_analysis.json" ;;
  esac
  if [ -f "$out" ]; then echo "skip $cmd $tag"; continue; fi
  echo "START $cmd $tag $(date +%H:%M:%S)"
  python3 run_experiments.py $line > "results/logs/${cmd}_$tag.log" 2>&1 \
    && echo "DONE $cmd $tag $(date +%H:%M:%S)" || echo "FAILED $cmd $tag $(date +%H:%M:%S)"
done < "$1"
