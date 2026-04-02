#!/usr/bin/env bash
# Run the 7k-population 100k-generation experiment on the instance.
# Auto-destroys instance when complete via VAST_INSTANCE_ID + VAST_API_KEY env vars.
# Usage: VAST_INSTANCE_ID=<id> VAST_API_KEY=<key> bash run-7k.sh
set -euo pipefail

RUN_NAME="7k-seed42"
RESULTS_DIR="/workspace/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

echo "=== Starting $RUN_NAME ==="
echo "Instance: ${VAST_INSTANCE_ID:-unknown}"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$RESULTS_DIR"

python3 -m semgpu.main 42 100000 \
  --pop 7000 \
  --grid 178 \
  --food 1050 \
  --zone-radius 71 \
  --zone-speed 4.45 \
  --zone-drain 0.15 \
  --signal-cost 0.015 \
  --signal-range 71 \
  --ticks 500 \
  --pred 5 \
  --freeze-zones 2 \
  --poison-ratio 0.0 \
  --metrics-interval 10 \
  --checkpoint-interval 1000

EXIT_CODE=$?
echo "Python exited with code $EXIT_CODE at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Write completion marker (local cron monitor detects this)
echo "{\"status\":\"complete\",\"exit_code\":$EXIT_CODE,\"timestamp\":\"$(date -Iseconds)\"}" \
  > /workspace/results/done.json

echo "=== Experiment done, triggering teardown ==="
# Self-destruct via Vast.ai API
if [ -n "${VAST_API_KEY:-}" ] && [ -n "${VAST_INSTANCE_ID:-}" ]; then
    curl -s -X DELETE \
      "https://console.vast.ai/api/v0/instances/${VAST_INSTANCE_ID}/?api_key=${VAST_API_KEY}" \
      && echo "Instance destroy request sent." \
      || echo "WARNING: destroy API call failed - manual cleanup needed"
else
    echo "WARNING: VAST_INSTANCE_ID or VAST_API_KEY not set - manual cleanup needed"
fi
