# semiotic-emergence-gpu

GPU-optimized evolutionary simulation exploring emergent communication. Python + JAX reimagining of [semiotic-emergence](https://github.com/onblueroses/semiotic-emergence), targeting 100k prey on cloud GPUs.

## Setup

```bash
# CPU (development/testing)
pip install -e ".[dev]"

# GPU (CUDA 12, Linux)
pip install -e ".[dev]"
pip install "jax[cuda12-local]"

# Or use the setup script on a fresh GPU VPS:
./setup-gpu.sh
```

## Usage

```bash
# Single run
python -m semgpu.main <seed> <generations> [flags]

# Batch mode (multiple seeds + divergence matrix)
python -m semgpu.main --batch <n_seeds> <generations> [flags]

# Via launch script (isolated run directory, background process)
./launch.sh <name> <seed> <generations> [flags]
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--pop` | 384 | Population size |
| `--grid` | 56 | Grid dimension |
| `--pred` | 3 | Number of kill zones |
| `--food` | 100 | Food count |
| `--ticks` | 500 | Ticks per generation |
| `--zone-radius` | 8.0 | Kill zone radius |
| `--zone-speed` | 0.5 | Zone movement probability |
| `--zone-drain` | 0.02 | Zone drain rate |
| `--signal-cost` | 0.002 | Energy cost per signal |
| `--signal-range` | auto | Signal reception range |
| `--signal-ticks` | 4 | Signal persistence |
| `--patch-ratio` | 0.5 | Fraction of cooperative food |
| `--kin-bonus` | 0.1 | Kin selection strength |
| `--no-signals` | false | Counterfactual: disable signaling |
| `--zone-coverage` | - | Auto-scale zones by area fraction |

## Output

- `output.csv` - 23 columns: fitness, MI, JSD, iconicity, entropy, etc.
- `trajectory.csv` - 47 columns: contingency matrix, per-symbol JSD, contrast
- `input_mi.csv` - 37 columns: MI between each input dimension and emitted symbol
- `divergence.csv` - NxN cross-population JSD matrix (batch mode only)

## Tests

```bash
python -m pytest tests/ -v
```
