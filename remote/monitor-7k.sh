#!/usr/bin/env bash
# Local cron-based monitor for the 7k run on Vast.ai instance 34012258.
# Runs every 10 minutes via cron. Copies results and destroys instance on completion.
#
# Install: crontab -e
# */10 * * * * /home/onblueroses/Work/semiotic-emergence-gpu/remote/monitor-7k.sh >> /home/onblueroses/Work/semiotic-emergence-gpu/remote/monitor.log 2>&1
set -euo pipefail

INSTANCE_ID="34012705"
LOCAL_RESULTS="/home/onblueroses/Work/semiotic-emergence-gpu/remote/results-7k"
DONE_MARKER="$LOCAL_RESULTS/done.json"
THIS_SCRIPT="$(realpath "$0")"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Checking instance $INSTANCE_ID..."

# Check if instance is still running
INSTANCE_STATUS=$(vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys
instances = json.load(sys.stdin)
for i in instances:
    if str(i.get('id')) == '$INSTANCE_ID':
        print(i.get('actual_status', 'unknown'))
        sys.exit(0)
print('not_found')
" 2>/dev/null || echo "error")

echo "  Instance status: $INSTANCE_STATUS"

if [ "$INSTANCE_STATUS" = "not_found" ] || [ "$INSTANCE_STATUS" = "error" ]; then
    echo "  Instance gone or unreachable - checking local results."
    if [ -f "$DONE_MARKER" ]; then
        echo "  Done marker exists - removing cron entry."
        # Remove this script's cron entry
        crontab -l 2>/dev/null | grep -v "$THIS_SCRIPT" | crontab - 2>/dev/null || true
        echo "  Monitor complete."
    fi
    exit 0
fi

# Try to fetch done marker from instance
mkdir -p "$LOCAL_RESULTS"
vastai copy "C.${INSTANCE_ID}:/workspace/results/done.json" "$LOCAL_RESULTS/" 2>/dev/null || true

if [ -f "$DONE_MARKER" ]; then
    echo "  Done marker found! Fetching results..."

    # Copy all results
    vastai copy "C.${INSTANCE_ID}:/workspace/results/7k-seed42/" "$LOCAL_RESULTS/7k-seed42/" \
      && echo "  Results copied." \
      || echo "  WARNING: Result copy failed - instance may already be destroyed."

    # Destroy instance
    vastai destroy instance "$INSTANCE_ID" \
      && echo "  Instance destroyed." \
      || echo "  WARNING: Destroy failed - check manually."

    # Remove cron entry
    crontab -l 2>/dev/null | grep -v "$THIS_SCRIPT" | crontab - 2>/dev/null || true
    echo "  Monitor complete - cron entry removed."
else
    # Periodic progress sync
    vastai copy "C.${INSTANCE_ID}:/workspace/results/7k-seed42/results.csv" \
      "$LOCAL_RESULTS/results-partial.csv" 2>/dev/null \
      && echo "  Progress CSV synced." \
      || echo "  No CSV yet."
fi
