#!/usr/bin/env bash
# Quick validation: run a short simulation and check output sanity.
# Usage: ./validate.sh [seed] [generations]
# Default: seed=42, 20 generations, default params (384 pop, 56 grid)

set -euo pipefail

SEED="${1:-42}"
GENS="${2:-20}"
WORKDIR=$(mktemp -d)

echo "=== Validation Run ==="
echo "Seed: $SEED  Gens: $GENS  Dir: $WORKDIR"
echo ""

cd "$WORKDIR"

# Run simulation
python3 -m semgpu.main "$SEED" "$GENS" 2>&1 | tee run.log

echo ""
echo "=== Output Checks ==="

PASS=0
FAIL=0

check() {
    local desc="$1" result="$2"
    if [ "$result" = "true" ]; then
        echo "  [OK] $desc"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $desc"
        FAIL=$((FAIL + 1))
    fi
}

# Check files exist and have content
for f in output.csv trajectory.csv input_mi.csv; do
    EXISTS=$([ -f "$f" ] && [ -s "$f" ] && echo "true" || echo "false")
    check "$f exists and non-empty" "$EXISTS"
done

# Check column counts
OUT_COLS=$(head -1 output.csv | awk -F, '{print NF}')
check "output.csv has 23 columns (got $OUT_COLS)" "$([ "$OUT_COLS" = "23" ] && echo true || echo false)"

TRAJ_COLS=$(head -1 trajectory.csv | awk -F, '{print NF}')
check "trajectory.csv has 47 columns (got $TRAJ_COLS)" "$([ "$TRAJ_COLS" = "47" ] && echo true || echo false)"

IMI_COLS=$(head -1 input_mi.csv | awk -F, '{print NF}')
check "input_mi.csv has 37 columns (got $IMI_COLS)" "$([ "$IMI_COLS" = "37" ] && echo true || echo false)"

# Check row counts (header + GENS data rows)
EXPECTED_ROWS=$((GENS + 1))
OUT_ROWS=$(wc -l < output.csv)
check "output.csv has $EXPECTED_ROWS rows (got $OUT_ROWS)" "$([ "$OUT_ROWS" = "$EXPECTED_ROWS" ] && echo true || echo false)"

# Check fitness is positive and rising
python3 -c "
import csv, sys
with open('output.csv') as f:
    rows = list(csv.DictReader(f))
first_fit = float(rows[0]['avg_fitness'])
last_fit = float(rows[-1]['avg_fitness'])
first_zd = int(rows[0]['zone_deaths'])
last_sig = int(rows[-1]['signals_emitted'])
print(f'  Fitness: {first_fit:.1f} -> {last_fit:.1f}')
print(f'  Zone deaths (gen 0): {first_zd}')
print(f'  Signals (last gen): {last_sig}')
# Sanity: last fitness should be > 0
if last_fit <= 0:
    print('  [FAIL] Final fitness <= 0')
    sys.exit(1)
# Fitness should generally increase (not guaranteed in 20 gens, but basic sanity)
if last_fit > first_fit * 0.5:
    print('  [OK] Fitness not collapsing')
else:
    print('  [WARN] Fitness may be declining significantly')
"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

# Cleanup
rm -rf "$WORKDIR"

if [ "$FAIL" -gt 0 ]; then
    echo "VALIDATION FAILED"
    exit 1
else
    echo "VALIDATION PASSED"
fi
