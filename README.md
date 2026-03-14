# semiotic-emergence-gpu

GPU-optimized evolutionary simulation exploring emergent communication. Python + JAX reimagining of [semiotic-emergence](https://github.com/onblueroses/semiotic-emergence), targeting 100k prey on cloud GPUs.

## Setup

```bash
# CPU (development)
pip install -e ".[cpu,dev]"

# CUDA 12 (cloud GPU)
pip install -e ".[dev]"
```

## Usage

```bash
python -m semgpu.main <seed> <generations>
```
