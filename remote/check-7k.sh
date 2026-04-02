#!/usr/bin/env bash
# Check progress of the 7k run on instance 34012705.
# Usage: bash remote/check-7k.sh
set -euo pipefail
INSTANCE_ID="34012705"

echo "=== Instance status ==="
vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('  status:  ', d.get('actual_status'))
print('  $/hr:    ', d.get('dph_total'))
print('  uptime:  ', d.get('duration', 0), 's')
"

echo ""
echo "=== Run log (last 30 lines) ==="
vastai copy "C.${INSTANCE_ID}:/workspace/run.log" /tmp/semgpu-run.log 2>/dev/null \
  && tail -30 /tmp/semgpu-run.log \
  || echo "  (log not available yet)"

echo ""
echo "=== CSV progress ==="
vastai copy "C.${INSTANCE_ID}:/workspace/results/7k-seed42/output.csv" /tmp/semgpu-results.csv 2>/dev/null \
  && echo "  Lines: $(wc -l < /tmp/semgpu-results.csv) (header + generations)" \
  && tail -3 /tmp/semgpu-results.csv \
  || echo "  (CSV not available yet)"
