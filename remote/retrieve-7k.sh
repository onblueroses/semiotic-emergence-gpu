#!/usr/bin/env bash
# Retrieve results from instance 34012705 and (optionally) destroy it.
# Usage: bash remote/retrieve-7k.sh [--destroy]
set -euo pipefail
INSTANCE_ID="34012705"
LOCAL_RESULTS="remote/results-7k"

mkdir -p "$LOCAL_RESULTS"

echo "=== Copying results ==="
vastai copy "C.${INSTANCE_ID}:/workspace/results/7k-seed42/" "$LOCAL_RESULTS/7k-seed42/" \
  && echo "Done: $LOCAL_RESULTS/7k-seed42/" \
  || echo "ERROR: copy failed"

vastai copy "C.${INSTANCE_ID}:/workspace/run.log" "$LOCAL_RESULTS/run.log" 2>/dev/null || true

if [ "${1:-}" = "--destroy" ]; then
    echo ""
    echo "=== Destroying instance ==="
    vastai destroy instance "$INSTANCE_ID" && echo "Instance destroyed."
fi
