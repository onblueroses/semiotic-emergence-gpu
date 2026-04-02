#!/usr/bin/env bash
# Run once on a fresh Vast.ai CUDA instance to set up semgpu.
# Usage: bash setup.sh
set -euo pipefail

echo "=== Installing JAX (CUDA 12) ==="
pip install --quiet --upgrade \
  "jax[cuda12_pip]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "=== Cloning repo ==="
git clone https://github.com/onblueroses/semiotic-emergence-gpu.git /workspace/semgpu

echo "=== Installing package ==="
cd /workspace/semgpu
pip install --quiet -e .

echo "=== Verifying GPU ==="
python3 -c "
import jax
devs = jax.devices()
print('JAX devices:', devs)
assert devs[0].platform == 'gpu', f'Expected GPU, got {devs[0].platform}'
print('OK: JAX sees GPU')
"
