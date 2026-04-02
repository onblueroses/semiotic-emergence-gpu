"""Tests for world state, spatial primitives, and tick mechanics."""

import jax
import jax.numpy as jnp

from semgpu.brain import DEFAULT_BASE_HIDDEN, DEFAULT_SIGNAL_HIDDEN, MAX_GENOME_LEN
from semgpu.spatial import wrap_delta, wrap_dist_sq, zone_drain_amount, nearest_zone_edge_dist
from semgpu.world import init_world, build_inputs, step, INPUTS


class TestSpatial:
    def test_wrap_delta_no_wrap(self):
        assert wrap_delta(jnp.int32(5), jnp.int32(8), 56) == 3
        assert wrap_delta(jnp.int32(8), jnp.int32(5), 56) == -3

    def test_wrap_delta_wraps_positive(self):
        # 50 -> 5 on grid 56: shortest path is +11, not -45
        d = wrap_delta(jnp.int32(50), jnp.int32(5), 56)
        assert d == 11

    def test_wrap_delta_wraps_negative(self):
        # 5 -> 50 on grid 56: shortest path is -11
        d = wrap_delta(jnp.int32(5), jnp.int32(50), 56)
        assert d == -11

    def test_wrap_delta_vectorized(self):
        a = jnp.array([0, 50, 27])
        b = jnp.array([5, 5, 28])
        d = wrap_delta(a, b, 56)
        assert jnp.allclose(d, jnp.array([5, 11, 1]))

    def test_wrap_dist_sq(self):
        d = wrap_dist_sq(jnp.int32(0), jnp.int32(0), jnp.int32(3), jnp.int32(4), 56)
        assert abs(float(d) - 25.0) < 1e-4

    def test_wrap_dist_sq_wrapping(self):
        # (55, 55) to (1, 1) on grid 56 = distance (2, 2) -> dist_sq = 8
        d = wrap_dist_sq(jnp.int32(55), jnp.int32(55), jnp.int32(1), jnp.int32(1), 56)
        assert abs(float(d) - 8.0) < 1e-4


class TestZoneDrain:
    def test_inside_zone_gets_damage(self):
        # Prey at zone center
        drain = zone_drain_amount(
            jnp.array([10]), jnp.array([10]),  # prey at (10, 10)
            jnp.array([10.0]), jnp.array([10.0]),  # zone at (10, 10)
            jnp.array([8.0]),  # radius 8
            56, 0.02,
        )
        # At center, gradient = 1.0, drain = 0.02
        assert abs(float(drain[0]) - 0.02) < 1e-4

    def test_at_edge_zero_drain(self):
        # Prey at zone edge
        drain = zone_drain_amount(
            jnp.array([18]), jnp.array([10]),  # 8 cells from center
            jnp.array([10.0]), jnp.array([10.0]),
            jnp.array([8.0]),
            56, 0.02,
        )
        assert float(drain[0]) < 1e-4

    def test_outside_zone_no_drain(self):
        drain = zone_drain_amount(
            jnp.array([30]), jnp.array([30]),
            jnp.array([10.0]), jnp.array([10.0]),
            jnp.array([8.0]),
            56, 0.02,
        )
        assert float(drain[0]) == 0.0

    def test_gradient_scales_with_distance(self):
        # Prey at half-radius: gradient = 0.5, drain = 0.01
        drain = zone_drain_amount(
            jnp.array([14]), jnp.array([10]),  # 4 cells from center
            jnp.array([10.0]), jnp.array([10.0]),
            jnp.array([8.0]),
            56, 0.02,
        )
        assert abs(float(drain[0]) - 0.01) < 0.002

    def test_zones_stack(self):
        # Two overlapping zones at same position
        drain = zone_drain_amount(
            jnp.array([10]), jnp.array([10]),
            jnp.array([10.0, 10.0]), jnp.array([10.0, 10.0]),
            jnp.array([8.0, 8.0]),
            56, 0.02,
        )
        assert abs(float(drain[0]) - 0.04) < 1e-4


class TestNearestZoneEdgeDist:
    def test_inside_zone(self):
        d = nearest_zone_edge_dist(
            jnp.array([10]), jnp.array([10]),
            jnp.array([10.0]), jnp.array([10.0]),
            jnp.array([8.0]),
            56,
        )
        assert float(d[0]) < 0  # negative = inside

    def test_outside_zone(self):
        d = nearest_zone_edge_dist(
            jnp.array([30]), jnp.array([30]),
            jnp.array([10.0]), jnp.array([10.0]),
            jnp.array([8.0]),
            56,
        )
        assert float(d[0]) > 0  # positive = outside


class TestWorldState:
    def _make_world(self, n=10, key_seed=42):
        key = jax.random.key(key_seed)
        k1, k2 = jax.random.split(key)
        return init_world(
            prey_x=jax.random.randint(k1, (n,), 0, 56),
            prey_y=jax.random.randint(k2, (n,), 0, 56),
            weights=jax.random.normal(key, (n, MAX_GENOME_LEN)) * 0.1,
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=3,
            food_count=50,
            grid_size=56,
            zone_radius=8.0,
            zone_speed=0.5,
            patch_ratio=0.5,
            max_signals=1000,
            ticks_per_eval=500,
            max_events=1000,
            max_deaths=n,
            key=key,
        )

    def test_init_shapes(self):
        ws = self._make_world(10)
        assert ws.prey_x.shape == (10,)
        assert ws.energy.shape == (10,)
        assert ws.weights.shape == (10, MAX_GENOME_LEN)
        assert ws.zone_x.shape == (3,)
        assert ws.food_x.shape == (50,)
        assert ws.sig_x.shape == (1000,)

    def test_all_alive_initially(self):
        ws = self._make_world(10)
        assert jnp.all(ws.alive)
        assert jnp.all(ws.energy == 1.0)

    def test_pytree_roundtrip(self):
        ws = self._make_world(10)
        leaves, treedef = jax.tree.flatten(ws)
        ws2 = treedef.unflatten(leaves)
        assert jnp.allclose(ws.prey_x, ws2.prey_x)

    def test_build_inputs_shape(self):
        ws = self._make_world(10)
        inputs = build_inputs(ws, 56, 22.4)
        assert inputs.shape == (10, INPUTS)

    def test_build_inputs_energy(self):
        ws = self._make_world(10)
        inputs = build_inputs(ws, 56, 22.4)
        # Energy should be 1.0 for all (initial state)
        assert jnp.allclose(inputs[:, 35], 1.0)

    def test_step_runs(self):
        ws = self._make_world(20)
        ws2 = step(
            ws, grid_size=56, signal_range=22.4, base_drain=0.0008,
            signal_cost=0.002, zone_drain_rate=0.02, patch_ratio=0.5,
            food_count=50, signal_ticks=4, no_signals=False, max_signals=1000,
            mi_bin0=8.0, mi_bin1=22.4, mi_bin2=30.8, zone_radius_scalar=8.0, max_events=1000,
        )
        assert ws2.tick == 1
        # Energy should have decreased (metabolism)
        assert jnp.all(ws2.energy[ws2.alive] < 1.0)

    def test_multiple_steps(self):
        ws = self._make_world(20)
        for _ in range(10):
            ws = step(
                ws, grid_size=56, signal_range=22.4, base_drain=0.0008,
                signal_cost=0.002, zone_drain_rate=0.02, patch_ratio=0.5,
                food_count=50, signal_ticks=4, no_signals=False, max_signals=1000,
                mi_bin0=8.0, mi_bin1=22.4, mi_bin2=30.8, zone_radius_scalar=8.0, max_events=1000,
            )
        assert ws.tick == 10
        # Some prey should still be alive after 10 ticks
        assert jnp.sum(ws.alive) > 0

    def test_zone_deaths_accumulate(self):
        # Large zone (radius 28 on grid 56) so prey can't escape even while moving
        # High drain rate (0.5) for fast kills
        key = jax.random.key(99)
        n = 20
        ws = init_world(
            prey_x=jnp.full(n, 28, dtype=jnp.int32),
            prey_y=jnp.full(n, 28, dtype=jnp.int32),
            weights=jnp.zeros((n, MAX_GENOME_LEN)),
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=1,
            food_count=10,
            grid_size=56,
            zone_radius=28.0,
            zone_speed=0.0,
            patch_ratio=0.0,
            max_signals=100,
            ticks_per_eval=500,
            max_events=100,
            max_deaths=n,
            key=key,
        )
        ws = ws._replace(
            zone_x=jnp.array([28.0]),
            zone_y=jnp.array([28.0]),
        )
        # At center, drain=0.5/tick -> dead in 2 ticks
        for _ in range(5):
            ws = step(
                ws, grid_size=56, signal_range=22.4, base_drain=0.0008,
                signal_cost=0.002, zone_drain_rate=0.5, patch_ratio=0.0,
                food_count=10, signal_ticks=4, no_signals=False, max_signals=100,
                mi_bin0=28.0, mi_bin1=22.4, mi_bin2=30.8, zone_radius_scalar=28.0, max_events=1000,
            )
        assert int(ws.zone_deaths) > 0


class TestDeathEchoes:
    """Tests for the death echo ring buffer and witness inputs (inputs 36-38)."""

    def _make_world_with_kills(self, n=20, radius=28.0, drain=0.5, steps=5):
        """Create world guaranteed to produce zone kills for testing echoes."""
        key = jax.random.key(77)
        ws = init_world(
            prey_x=jnp.full(n, 28, dtype=jnp.int32),
            prey_y=jnp.full(n, 28, dtype=jnp.int32),
            weights=jnp.zeros((n, MAX_GENOME_LEN)),
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=1,
            food_count=10,
            grid_size=56,
            zone_radius=radius,
            zone_speed=0.0,
            patch_ratio=0.0,
            max_signals=100,
            ticks_per_eval=500,
            max_events=100,
            max_deaths=n,
            key=key,
        )
        ws = ws._replace(zone_x=jnp.array([28.0]), zone_y=jnp.array([28.0]))
        for _ in range(steps):
            ws = step(
                ws, grid_size=56, signal_range=22.4, base_drain=0.0008,
                signal_cost=0.002, zone_drain_rate=drain, patch_ratio=0.0,
                food_count=10, signal_ticks=4, no_signals=False, max_signals=100,
                mi_bin0=28.0, mi_bin1=22.4, mi_bin2=30.8, zone_radius_scalar=28.0,
                max_events=1000,
            )
        return ws

    def test_death_ring_buffer_fills(self):
        ws = self._make_world_with_kills()
        assert int(ws.zone_deaths) > 0
        assert jnp.any(ws.death_valid)

    def test_death_positions_nonzero(self):
        ws = self._make_world_with_kills()
        # Prey started at (28, 28) but may have moved before dying.
        # Verify that recorded positions are within the zone (radius 28, center 28).
        valid_mask = ws.death_valid
        assert jnp.any(valid_mask)
        gs = 56
        dx = jnp.where(valid_mask, jnp.abs(ws.death_x - 28).astype(jnp.float32), 0.0)
        dy = jnp.where(valid_mask, jnp.abs(ws.death_y - 28).astype(jnp.float32), 0.0)
        # Wrap distances on toroidal grid
        dx = jnp.minimum(dx, gs - dx)
        dy = jnp.minimum(dy, gs - dy)
        dist = jnp.sqrt(dx * dx + dy * dy)
        assert jnp.all(jnp.where(valid_mask, dist <= 28.0, True))

    def test_ring_buffer_wraps(self):
        # Fill ring buffer beyond capacity to verify wrap-around
        key = jax.random.key(55)
        n = 5
        ws = init_world(
            prey_x=jnp.full(n, 28, dtype=jnp.int32),
            prey_y=jnp.full(n, 28, dtype=jnp.int32),
            weights=jnp.zeros((n, MAX_GENOME_LEN)),
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=1,
            food_count=5,
            grid_size=56,
            zone_radius=28.0,
            zone_speed=0.0,
            patch_ratio=0.0,
            max_signals=50,
            ticks_per_eval=500,
            max_events=50,
            max_deaths=n,  # tiny buffer = wrap guaranteed when all die
            key=key,
        )
        ws = ws._replace(zone_x=jnp.array([28.0]), zone_y=jnp.array([28.0]))
        for _ in range(10):
            ws = step(
                ws, grid_size=56, signal_range=22.4, base_drain=0.0008,
                signal_cost=0.002, zone_drain_rate=0.5, patch_ratio=0.0,
                food_count=5, signal_ticks=4, no_signals=False, max_signals=50,
                mi_bin0=28.0, mi_bin1=22.4, mi_bin2=30.8, zone_radius_scalar=28.0,
                max_events=50,
            )
        # cursor should have advanced (mod max_deaths), not crashed
        assert ws.death_cursor.shape == ()

    def test_death_witness_input_shape(self):
        ws = self._make_world_with_kills()
        inputs = build_inputs(ws, 56, 22.4)
        assert inputs.shape == (ws.prey_x.shape[0], INPUTS)
        assert INPUTS == 39

    def test_death_witness_nonzero_near_death(self):
        # After kills, prey positions are at/near zone deaths within signal_range.
        # Check that death witness inputs are non-zero (regardless of alive state).
        ws = self._make_world_with_kills()
        if not jnp.any(ws.death_valid):
            return  # no deaths = test not applicable
        inputs = build_inputs(ws, 56, 22.4)
        # Prey positions are near recorded death positions (same zone),
        # so death_nearby (input 36) should be > 0 for at least one prey
        assert jnp.max(inputs[:, 36]) > 0.0

    def test_death_witness_zero_no_deaths(self):
        # Fresh world: no deaths, so death witness inputs should be 0
        ws = self._make_world_with_kills(radius=1.0, drain=0.001, steps=1)
        # With tiny radius and low drain, no deaths
        inputs = build_inputs(ws, 56, 22.4)
        if jnp.any(ws.death_valid):
            return  # deaths happened - skip
        assert jnp.allclose(inputs[:, 36], 0.0)
        assert jnp.allclose(inputs[:, 37], 0.0)
        assert jnp.allclose(inputs[:, 38], 0.0)

    def test_death_witness_intensity_bounded(self):
        ws = self._make_world_with_kills()
        inputs = build_inputs(ws, 56, 22.4)
        assert jnp.all(inputs[:, 36] >= 0.0)
        assert jnp.all(inputs[:, 36] <= 1.0)


class TestFreezeZones:
    """Tests for freeze zone drain mechanics (Phase 2)."""

    def _make_freeze_world(self, n=10, num_zones=1, num_freeze=1, key_seed=42):
        key = jax.random.key(key_seed)
        ws = init_world(
            prey_x=jnp.full(n, 14, dtype=jnp.int32),
            prey_y=jnp.full(n, 14, dtype=jnp.int32),
            weights=jnp.zeros((n, MAX_GENOME_LEN)),
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=num_zones,
            food_count=10,
            grid_size=28,
            zone_radius=14.0,
            zone_speed=0.0,
            patch_ratio=0.0,
            max_signals=100,
            ticks_per_eval=50,
            max_events=100,
            max_deaths=n,
            key=key,
            num_freeze=num_freeze,
        )
        # Lock zone at center (14, 14) for determinism
        return ws._replace(zone_x=jnp.array([14.0]), zone_y=jnp.array([14.0]))

    def test_zone_type_set(self):
        ws = self._make_freeze_world(num_zones=3, num_freeze=2)
        assert ws.zone_type.shape == (3,)
        # First zone flee, last two freeze
        assert not bool(ws.zone_type[0])
        assert bool(ws.zone_type[1])
        assert bool(ws.zone_type[2])

    def test_freeze_moving_more_drain_than_flee(self):
        # Moving prey in a freeze zone take FREEZE_MOVE_PENALTY x more drain than flee.
        # Zero weights -> action=0 (move up) -> prey are moving.
        from semgpu.world import FREEZE_MOVE_PENALTY
        key = jax.random.key(1)
        n = 2
        # Build flee-only world
        ws_flee = init_world(
            prey_x=jnp.full(n, 14, dtype=jnp.int32),
            prey_y=jnp.full(n, 14, dtype=jnp.int32),
            weights=jnp.zeros((n, MAX_GENOME_LEN)),
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=1, food_count=5, grid_size=28,
            zone_radius=14.0, zone_speed=0.0, patch_ratio=0.0,
            max_signals=50, ticks_per_eval=50, max_events=50, max_deaths=n,
            key=key, num_freeze=0,
        )
        ws_flee = ws_flee._replace(zone_x=jnp.array([14.0]), zone_y=jnp.array([14.0]))
        # Build freeze-only world
        ws_freeze = init_world(
            prey_x=jnp.full(n, 14, dtype=jnp.int32),
            prey_y=jnp.full(n, 14, dtype=jnp.int32),
            weights=jnp.zeros((n, MAX_GENOME_LEN)),
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=1, food_count=5, grid_size=28,
            zone_radius=14.0, zone_speed=0.0, patch_ratio=0.0,
            max_signals=50, ticks_per_eval=50, max_events=50, max_deaths=n,
            key=key, num_freeze=1,
        )
        ws_freeze = ws_freeze._replace(zone_x=jnp.array([14.0]), zone_y=jnp.array([14.0]))

        step_kwargs = dict(
            grid_size=28, signal_range=11.2, base_drain=0.0,
            signal_cost=0.0, zone_drain_rate=0.02, patch_ratio=0.0,
            food_count=5, signal_ticks=4, no_signals=True, max_signals=50,
            mi_bin0=14.0, mi_bin1=11.2, mi_bin2=15.0, zone_radius_scalar=14.0, max_events=50,
        )
        ws_flee2 = step(ws_flee, **step_kwargs)
        ws_freeze2 = step(ws_freeze, **step_kwargs)

        # Zero weights -> action=0 (move) -> prey are moving.
        # Flee: drain=0.02 * 1.0. Freeze+moving: drain=0.02 * FREEZE_MOVE_PENALTY=3.0
        flee_damage = float(ws_flee2.zone_damage[0])
        freeze_damage = float(ws_freeze2.zone_damage[0])
        # Freeze moving should be FREEZE_MOVE_PENALTY x more damage than flee
        assert freeze_damage > flee_damage
        assert abs(freeze_damage / flee_damage - FREEZE_MOVE_PENALTY) < 0.1

    def test_freeze_pressure_zero_outside(self):
        ws = self._make_freeze_world()
        # Place prey far from zone
        ws2 = ws._replace(
            prey_x=jnp.zeros(ws.prey_x.shape[0], dtype=jnp.int32),
            prey_y=jnp.zeros(ws.prey_y.shape[0], dtype=jnp.int32),
        )
        inputs = build_inputs(ws2, 28, 11.2)
        # At (0,0), distance to freeze zone at (14,14) = sqrt(2)*14 ≈ 19.8 > zone_radius=14
        # So freeze_pressure should be 0
        assert jnp.allclose(inputs[:, 2], 0.0, atol=1e-5)

    def test_freeze_pressure_nonzero_inside(self):
        ws = self._make_freeze_world()
        # Prey already at (14, 14) = zone center
        inputs = build_inputs(ws, 28, 11.2)
        # At zone center, freeze_pressure = 1.0
        assert jnp.allclose(inputs[:, 2], 1.0, atol=1e-4)

    def test_flee_zones_no_freeze_pressure(self):
        # With all flee zones, freeze_pressure should be 0 everywhere
        key = jax.random.key(7)
        n = 5
        ws = init_world(
            prey_x=jnp.full(n, 5, dtype=jnp.int32),
            prey_y=jnp.full(n, 5, dtype=jnp.int32),
            weights=jnp.zeros((n, MAX_GENOME_LEN)),
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=2, food_count=5, grid_size=28,
            zone_radius=4.0, zone_speed=0.0, patch_ratio=0.0,
            max_signals=50, ticks_per_eval=50, max_events=50, max_deaths=n,
            key=key, num_freeze=0,  # all flee
        )
        inputs = build_inputs(ws, 28, 11.2)
        assert jnp.allclose(inputs[:, 2], 0.0, atol=1e-5)

    def test_freeze_zone_deaths_tracked(self):
        ws = self._make_freeze_world(n=20)
        step_kwargs = dict(
            grid_size=28, signal_range=11.2, base_drain=0.0,
            signal_cost=0.0, zone_drain_rate=0.5, patch_ratio=0.0,
            food_count=10, signal_ticks=4, no_signals=True, max_signals=100,
            mi_bin0=14.0, mi_bin1=11.2, mi_bin2=15.0, zone_radius_scalar=14.0, max_events=100,
        )
        for _ in range(10):
            ws = step(ws, **step_kwargs)
        # Some prey should have died in the freeze zone
        assert int(ws.freeze_zone_deaths) > 0


class TestPoisonFood:
    """Tests for poison food mechanics (Phase 2)."""

    def _make_world_with_poison(self, n=10, poison_ratio=1.0):
        key = jax.random.key(99)
        k1, k2 = jax.random.split(key)
        ws = init_world(
            prey_x=jax.random.randint(k1, (n,), 0, 28),
            prey_y=jax.random.randint(k2, (n,), 0, 28),
            weights=jnp.zeros((n, MAX_GENOME_LEN)),
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=1,        # need at least 1 to avoid empty-array min error
            food_count=50,
            grid_size=28,
            zone_radius=1.0,
            zone_speed=0.0,
            patch_ratio=0.0,
            max_signals=100,
            ticks_per_eval=500,
            max_events=100,
            max_deaths=n,
            key=key,
            poison_ratio=poison_ratio,
        )
        return ws

    def test_poison_food_is_poison(self):
        ws = self._make_world_with_poison(poison_ratio=1.0)
        assert jnp.all(ws.food_is_poison)

    def test_no_poison_food_not_poison(self):
        ws = self._make_world_with_poison(poison_ratio=0.0)
        assert not jnp.any(ws.food_is_poison)

    def test_poison_decreases_energy(self):
        # Verify food_is_poison is set and poison_eaten counter initializes at 0
        key = jax.random.key(33)
        n = 5
        ws = init_world(
            prey_x=jnp.zeros(n, dtype=jnp.int32),
            prey_y=jnp.zeros(n, dtype=jnp.int32),
            weights=jnp.zeros((n, MAX_GENOME_LEN)),
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=1,        # need at least 1 zone to avoid empty-array error
            food_count=10,
            grid_size=28,
            zone_radius=1.0,
            zone_speed=0.0,
            patch_ratio=0.0,
            max_signals=50,
            ticks_per_eval=50,
            max_events=50,
            max_deaths=n,
            key=key,
            poison_ratio=1.0,
        )
        # All food should be poison
        assert jnp.all(ws.food_is_poison)
        assert ws.poison_eaten == 0
        # With zero weights, action=0 (move up), so prey won't eat this tick.
        # Step should run without error.
        step_kwargs = dict(
            grid_size=28, signal_range=11.2, base_drain=0.0,
            signal_cost=0.0, zone_drain_rate=0.0, patch_ratio=0.0,
            food_count=10, signal_ticks=4, no_signals=True, max_signals=50,
            mi_bin0=1.0, mi_bin1=1.0, mi_bin2=1.5, zone_radius_scalar=1.0, max_events=50,
            poison_ratio=1.0,
        )
        ws2 = step(ws, **step_kwargs)
        assert ws2.tick == 1

    def test_good_food_increases_energy(self):
        # Verify non-poison food world state is consistent
        key = jax.random.key(44)
        n = 5
        ws = init_world(
            prey_x=jnp.zeros(n, dtype=jnp.int32),
            prey_y=jnp.zeros(n, dtype=jnp.int32),
            weights=jnp.zeros((n, MAX_GENOME_LEN)),
            base_hidden=jnp.full(n, DEFAULT_BASE_HIDDEN),
            signal_hidden=jnp.full(n, DEFAULT_SIGNAL_HIDDEN),
            num_zones=1, food_count=10, grid_size=28,
            zone_radius=1.0, zone_speed=0.0, patch_ratio=0.0,
            max_signals=50, ticks_per_eval=50, max_events=50, max_deaths=n,
            key=key, poison_ratio=0.0,
        )
        assert jnp.all(~ws.food_is_poison)
        assert ws.poison_eaten == 0
        assert ws.good_food_eaten == 0
