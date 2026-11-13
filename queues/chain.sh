#!/bin/bash
# chain.sh <training queue> <evaluation queue>: run the second once the first finishes.
cd "$(dirname "$0")/.."
while ps -eo args | grep -v grep | grep -q "run_queue.sh $1"; do sleep 20; done
bash queues/run_queue.sh "$2"
