#!/usr/bin/env bash
# Launch a semiotic-emergence-gpu run with proper isolation.
# Usage: ./launch.sh <name> <seed> <generations> [extra flags...]
# Example: ./launch.sh baseline-seed42 42 1000
# Example: ./launch.sh scale-10k 0 500 --pop 10000 --grid 200 --food 2000

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="$REPO_DIR/runs"

# --- Validate args ---
if [ $# -lt 3 ]; then
    echo "Usage: $0 <name> <seed> <generations> [extra flags...]"
    echo ""
    echo "Examples:"
    echo "  $0 baseline-42 42 1000"
    echo "  $0 rust-equiv 42 1000 --pop 384 --grid 56 --ticks 500"
    echo "  $0 scale-10k 0 500 --pop 10000 --grid 200 --food 2000"
    echo "  $0 batch-3 0 200 --batch 3 200 --pop 384 --grid 56"
    exit 1
fi

NAME="$1"; SEED="$2"; GENS="$3"; shift 3
EXTRA_FLAGS="$*"

# --- Check run directory doesn't already exist ---
RUN_DIR="$RUNS_DIR/$NAME"
if [ -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory already exists: $RUN_DIR"
    echo "Pick a different name or remove the existing directory."
    exit 1
fi

# --- Create run directory ---
mkdir -p "$RUN_DIR"

# --- Detect JAX backend ---
JAX_BACKEND=$(python3 -c "import jax; print(jax.default_backend())" 2>/dev/null || echo "unknown")
JAX_DEVICES=$(python3 -c "import jax; devs=jax.devices(); print(f'{len(devs)}x {devs[0].platform}')" 2>/dev/null || echo "unknown")

# --- Write metadata ---
GIT_COMMIT=$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(cd "$REPO_DIR" && git diff --quiet 2>/dev/null && echo "clean" || echo "dirty")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "$RUN_DIR/meta.txt" <<EOF
name: $NAME
seed: $SEED
generations: $GENS
extra_flags: $EXTRA_FLAGS
jax_backend: $JAX_BACKEND
jax_devices: $JAX_DEVICES
git_commit: $GIT_COMMIT ($GIT_DIRTY)
started: $TIMESTAMP
command: python3 -m semgpu.main $SEED $GENS $EXTRA_FLAGS
EOF

echo "=== Run: $NAME ==="
echo "Dir:     $RUN_DIR"
echo "Seed:    $SEED"
echo "Gens:    $GENS"
echo "Flags:   $EXTRA_FLAGS"
echo "Backend: $JAX_BACKEND ($JAX_DEVICES)"
echo "Commit:  $GIT_COMMIT ($GIT_DIRTY)"

# --- Launch with correct CWD ---
cd "$RUN_DIR"
nohup python3 -m semgpu.main "$SEED" "$GENS" $EXTRA_FLAGS \
    >> run.log 2>&1 </dev/null &
PID=$!
echo "$PID" > pid.txt

# Verify it survived
sleep 3
if kill -0 "$PID" 2>/dev/null; then
    echo "PID:     $PID (running)"
    echo ""
    echo "Monitor: tail -f $RUN_DIR/run.log"
    echo "Stop:    kill $PID"
else
    echo "ERROR: Process died immediately. Check $RUN_DIR/run.log"
    tail -20 "$RUN_DIR/run.log" 2>/dev/null
    exit 1
fi
