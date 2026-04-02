"""Integration tests - full generation evaluation."""


import jax
import jax.numpy as jnp

from semgpu.brain import DEFAULT_BASE_HIDDEN, DEFAULT_SIGNAL_HIDDEN, MAX_GENOME_LEN
from semgpu.world import evaluate_generation, init_world


def test_evaluate_generation_runs():
    """Run a full generation with small population."""
    key = jax.random.key(42)
    n = 20
    k1, k2, k3 = jax.random.split(key, 3)

    ws = init_world(
        prey_x=jax.random.randint(k1, (n,), 0, 20),
        prey_y=jax.random.randint(k2, (n,), 0, 20),
        weights=jax.random.normal(k3, (n, MAX_GENOME_LEN)) * 0.5,
        base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
        signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
        num_zones=3,
        food_count=30,
        grid_size=20,
        zone_radius=4.0,
        zone_speed=0.5,
        patch_ratio=0.5,
        max_signals=500,
        ticks_per_eval=50,
        max_events=1000,
        max_deaths=n,
        key=key,
    )

    result = evaluate_generation(
        ws,
        grid_size=20,
        signal_range=8.0,
        base_drain=0.0008,
        signal_cost=0.002,
        zone_drain_rate=0.02,
        patch_ratio=0.5,
        food_count=30,
        signal_ticks=4,
        no_signals=False,
        max_signals=500,
        ticks_per_eval=50,
    )

    assert result.fitness.shape == (n,)
    assert jnp.all(result.fitness >= 0)
    # After 50 ticks, some prey should have survived and eaten food
    assert jnp.max(result.ticks_alive) > 0
    assert result.final_state.tick == 50


def test_evaluate_generation_fitness_formula():
    """Verify fitness = ticks_alive * (1 + food_eaten * 0.1)."""
    key = jax.random.key(7)
    n = 10
    k1, k2, k3 = jax.random.split(key, 3)

    ws = init_world(
        prey_x=jax.random.randint(k1, (n,), 0, 20),
        prey_y=jax.random.randint(k2, (n,), 0, 20),
        weights=jax.random.normal(k3, (n, MAX_GENOME_LEN)) * 0.5,
        base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
        signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
        num_zones=1,
        food_count=50,
        grid_size=20,
        zone_radius=2.0,
        zone_speed=0.0,
        patch_ratio=0.0,
        max_signals=200,
        ticks_per_eval=100,
        max_events=1000,
        max_deaths=n,
        key=key,
    )

    result = evaluate_generation(
        ws,
        grid_size=20,
        signal_range=8.0,
        base_drain=0.0008,
        signal_cost=0.002,
        zone_drain_rate=0.02,
        patch_ratio=0.0,
        food_count=50,
        signal_ticks=4,
        no_signals=False,
        max_signals=200,
        ticks_per_eval=100,
    )

    expected = result.ticks_alive.astype(jnp.float32) * (
        1.0 + result.food_eaten.astype(jnp.float32) * 0.1
    )
    assert jnp.allclose(result.fitness, expected)


def test_no_signals_mode():
    """Counterfactual mode: no signals emitted."""
    key = jax.random.key(123)
    n = 10
    k1, k2, k3 = jax.random.split(key, 3)

    ws = init_world(
        prey_x=jax.random.randint(k1, (n,), 0, 20),
        prey_y=jax.random.randint(k2, (n,), 0, 20),
        weights=jax.random.normal(k3, (n, MAX_GENOME_LEN)) * 0.5,
        base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
        signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
        num_zones=1,
        food_count=20,
        grid_size=20,
        zone_radius=3.0,
        zone_speed=0.0,
        patch_ratio=0.0,
        max_signals=100,
        ticks_per_eval=50,
        max_events=1000,
        max_deaths=n,
        key=key,
    )

    result = evaluate_generation(
        ws,
        grid_size=20,
        signal_range=8.0,
        base_drain=0.0008,
        signal_cost=0.002,
        zone_drain_rate=0.02,
        patch_ratio=0.0,
        food_count=20,
        signal_ticks=4,
        no_signals=True,
        max_signals=100,
        ticks_per_eval=50,
    )

    assert int(result.total_signals) == 0


# ---- Checkpoint roundtrip tests ----

def test_checkpoint_roundtrip():
    """Save a checkpoint at gen 3 and load it back, verify arrays identical."""
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from semgpu.config import SimParams  # noqa: PLC0415
    from semgpu.main import load_checkpoint, save_checkpoint  # noqa: PLC0415

    key = jax.random.key(55)
    k1, k2, k3 = jax.random.split(key, 3)
    n = 10

    weights = jax.random.normal(k1, (n, MAX_GENOME_LEN)) * 0.5
    base_hidden = jnp.full(n, DEFAULT_BASE_HIDDEN, dtype=jnp.int32)
    signal_hidden = jnp.full(n, DEFAULT_BASE_HIDDEN, dtype=jnp.int32)
    prey_x = jax.random.randint(k2, (n,), 0, 20)
    prey_y = jax.random.randint(k3, (n,), 0, 20)
    params = SimParams()

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = save_checkpoint(
            tmpdir, 3, weights, base_hidden, signal_hidden, prey_x, prey_y, params, key,
        )
        assert os.path.exists(npz_path)
        json_path = npz_path.replace(".npz", ".json")
        assert os.path.exists(json_path)

        gen, arrays, params_dict = load_checkpoint(npz_path)
        assert gen == 3
        assert np.allclose(arrays["weights"], np.asarray(weights))
        assert np.allclose(arrays["base_hidden"], np.asarray(base_hidden))
        assert np.allclose(arrays["prey_x"], np.asarray(prey_x))
        assert params_dict["pop_size"] == 384


def test_checkpoint_smoke_creates_files():
    """Run 5 gens with --checkpoint-interval 2, verify checkpoint files exist."""
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from semgpu.config import SimParams  # noqa: PLC0415
    from semgpu.main import run_seed  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_dir = os.getcwd()
        os.chdir(tmpdir)
        try:
            params = SimParams.from_cli(["0", "5", "--pop", "20", "--grid", "10",
                                         "--pred", "1", "--ticks", "10",
                                         "--checkpoint-interval", "2"])
            run_seed(42, 5, params)
            # Checkpoints at gen 1 (0-indexed: after gen 1, so gen=1) and gen 3
            ckpt_files = [f for f in os.listdir("checkpoints") if f.endswith(".npz")]
            assert len(ckpt_files) >= 1
        finally:
            os.chdir(orig_dir)
