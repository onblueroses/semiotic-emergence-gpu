#!/usr/bin/env bash
# Setup script for GPU VPS. Run once after cloning the repo.
# Assumes: Ubuntu/Debian with NVIDIA driver 525+ and CUDA 12 already installed.
#
# Usage: ./setup-gpu.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== semiotic-emergence-gpu: GPU setup ==="

# 1. Check Python
echo ""
echo "--- Python ---"
python3 --version || { echo "ERROR: python3 not found"; exit 1; }

# 2. Check NVIDIA driver
echo ""
echo "--- NVIDIA Driver ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. CUDA may not work."
fi

# 3. Create venv
echo ""
echo "--- Virtual Environment ---"
VENV="$REPO_DIR/.venv"
if [ -d "$VENV" ]; then
    echo "Existing venv found at $VENV"
else
    python3 -m venv "$VENV"
    echo "Created venv at $VENV"
fi
source "$VENV/bin/activate"

# 4. Install JAX with CUDA
echo ""
echo "--- Installing JAX + CUDA 12 ---"
pip install --upgrade pip
pip install -e "$REPO_DIR[dev]"
pip install --upgrade "jax[cuda12-local]"

# 5. Verify JAX sees GPU
echo ""
echo "--- JAX Backend Check ---"
python3 -c "
import jax
devices = jax.devices()
print(f'Backend: {jax.default_backend()}')
print(f'Devices: {len(devices)}')
for d in devices:
    print(f'  {d.platform}: {d}')
if jax.default_backend() != 'gpu':
    print('WARNING: JAX is NOT using GPU!')
    print('Check CUDA installation and LD_LIBRARY_PATH (should be unset).')
else:
    print('GPU confirmed.')
"

# 6. Run tests
echo ""
echo "--- Running Tests ---"
cd "$REPO_DIR"
python3 -m pytest tests/ -v --timeout=120 2>&1 | tail -10

# 7. Quick smoke test (tiny run)
echo ""
echo "--- Smoke Test (5 gens, tiny pop) ---"
cd /tmp
python3 -m semgpu.main 0 5 --pop 20 --grid 10 --ticks 10 --food 10 --pred 1
echo "Smoke test passed."

# 8. Summary
echo ""
echo "=== Setup Complete ==="
echo "Activate: source $VENV/bin/activate"
echo "Run:      cd $REPO_DIR && ./launch.sh <name> <seed> <gens> [flags]"
echo ""
echo "Recommended first run (Rust-equivalent validation):"
echo "  ./launch.sh rust-equiv 42 1000 --pop 384 --grid 56 --ticks 500"
echo ""
echo "Scale ladder:"
echo "  ./launch.sh scale-1k   0 200 --pop 1000  --grid 80  --food 300  --ticks 200"
echo "  ./launch.sh scale-10k  0 100 --pop 10000 --grid 200 --food 2000 --ticks 200"
echo "  ./launch.sh scale-100k 0 50  --pop 100000 --grid 600 --food 20000 --ticks 100"
