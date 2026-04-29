"""
pytest for simulation.py

Tests verify the simulation loop, termination conditions,
Body class, and preset configurations.

Run with:
    pytest tests/test_simulation.py -v
"""

import numpy as np
import pytest
from three_body.simulation import Body, run_simulation, get_preset

# ----- Fix -----
# Fixtures are shared test configurations injected into tests automatically by pytest

@pytest.fixture
def simple_bodies():
    """
    Three well-separated bodies with mild velocities
    Expected to complete the full simulation without collision or escape
    """
    return [
        Body(1.0, [-10.0,  0.0], [0.0,  0.5], "Body 1"),
        Body(1.0, [ 10.0,  0.0], [0.0, -0.5], "Body 2"),
        Body(1.0, [  0.0, 10.0], [0.5,  0.0], "Body 3"),
    ]


@pytest.fixture
def collision_bodies():
    """
    Two bodies placed within collision radius of each other
    Expected to trigger collision termination immediately
    """
    return [
        Body(1.0, [0.0,  0.0], [0.0, 0.0], "Body 1"),
        Body(1.0, [0.3,  0.0], [0.0, 0.0], "Body 2"),
        Body(1.0, [0.0, 50.0], [0.0, 0.0], "Body 3"),
    ]


@pytest.fixture
def escape_bodies():
    """
    One body given a large outward velocity sufficient to escape the boundary
    Expected to trigger escape termination
    """
    return [
        Body(1.0, [  0.0, 0.0], [ 0.0, 0.0], "Body 1"),
        Body(1.0, [ 0.0, 50.0], [ 0.0, 0.0], "Body 2"),
        Body(1.0, [0.0, -50.0], [200.0, 0.0], "Body 3"),
    ]

# ----- Body -----

class TestBody:

    def test_stores_attributes_correctly(self):
        """
        Body must store mass, position, velocity and name
        exactly as provided on construction
        """
        b = Body(2.5, [1.0, -3.0], [0.5, 0.1], "Test")

        assert b.mass == 2.5
        assert np.allclose(b.position, [1.0, -3.0])
        assert np.allclose(b.velocity, [0.5, 0.1])
        assert b.name == "Test"

    def test_position_and_velocity_are_numpy_arrays(self):
        """
        Position and velocity must be stored as numpy arrays
        """
        b = Body(1.0, [1.0, 2.0], [0.0, 0.0])

        assert isinstance(b.position, np.ndarray)
        assert isinstance(b.velocity, np.ndarray)

# ----- SimulationStructure -----

class TestRunSimulationStructure:

    def test_returns_dict_with_all_keys(self, simple_bodies):
        """
        run_simulation must return a dict containing all expected keys
        """
        result = run_simulation(simple_bodies)
        expected_keys = {"positions", "masses", "names", "steps", "status", "message"}

        assert expected_keys == set(result.keys())

    def test_positions_shape(self, simple_bodies):
        """
        Trajectory array must be shape (n_steps, 3, 2)
        """
        result = run_simulation(simple_bodies)
        positions = result["positions"]

        assert positions.ndim == 3
        assert positions.shape[1] == 3
        assert positions.shape[2] == 2

    def test_steps_matches_trajectory_length(self, simple_bodies):
        """
        The step count must equal the number of frames
        in the trajectory array (must stay in sync for the
        visualizer to animate correctly)
        """
        result = run_simulation(simple_bodies)

        assert result["positions"].shape[0] == result["steps"]

    def test_masses_shape(self, simple_bodies):
        """
        Masses array must be shape (3,)
        """
        result = run_simulation(simple_bodies)

        assert result["masses"].shape == (3,)

    def test_names_match_input_bodies(self, simple_bodies):
        """
        Names in the result must match the names of the input Body instances
        """
        result = run_simulation(simple_bodies)

        assert len(result["names"]) == 3
        for i, name in enumerate(result["names"]):
            assert name == simple_bodies[i].name

# ----- TerminationConditions -----

class TestRunSimulationTermination:

    def test_stable_config_completes_normally(self, simple_bodies):
        """
        A stable configuration must report status 'completed',
        meaning no collision or escape was triggered.
        """
        result = run_simulation(simple_bodies)

        assert result["status"] == "completed"

    def test_collision_termination(self, collision_bodies):
        """
        Two bodies within collision radius must trigger status 'collision'
        The termination message must confirm which bodies collided
        """
        result = run_simulation(collision_bodies)

        assert result["status"] == "collision"
        assert "collided" in result["message"].lower()

    def test_escape_termination(self, escape_bodies):
        """
        A body with velocity sufficient to exceed the boundary must trigger status 'escape'
        The termination message must confirm which body escaped
        """
        result = run_simulation(escape_bodies)

        assert result["status"] == "escape"
        assert "escaped" in result["message"].lower()

    def test_collision_stops_simulation_early(self, collision_bodies):
        """
        A collision must stop the simulation before MAX_STEPS is reached
        """
        from three_body.simulation import MAX_STEPS
        result = run_simulation(collision_bodies)

        assert result["steps"] < MAX_STEPS


# ----- PhysicsChecksSimulation -----

class TestRunSimulationPhysics:

    def test_center_of_mass_at_origin(self, simple_bodies):
        """
        Trajectories are stored in the center of mass frame
        The mass-weighted average position at any frame must be
        at the origin within numerical precision
        """
        result = run_simulation(simple_bodies)
        positions = result["positions"]
        masses    = result["masses"]

        for frame_idx in [0, -1]:
            frame = positions[frame_idx]
            com = np.sum(masses[:, np.newaxis] * frame, axis=0) / np.sum(masses)
            assert np.allclose(com, 0.0, atol=1e-6)


# ----- Preset -----

class TestGetPreset:

    @pytest.mark.parametrize("name", ["figure8", "lagrange", "hierarchical"])
    def test_returns_three_body_instances(self, name):
        """
        Every preset must return exactly three Body instances
        """
        bodies = get_preset(name)

        assert len(bodies) == 3
        for b in bodies:
            assert isinstance(b, Body)


    def test_invalid_preset_raises_value_error(self):
        """
        get_preset must raise ValueError for an unrecognised name
        """
        with pytest.raises(ValueError):
            get_preset("nonexistent")

    @pytest.mark.parametrize("name", ["figure8", "lagrange", "hierarchical"])
    def test_preset_runs_more_than_ten_steps(self, name):
        """
        Every preset must simulate for more than 10 steps without
        immediately colliding or escaping.
        """
        bodies = get_preset(name)
        result = run_simulation(bodies)

        assert result["steps"] > 10, (
            f"Preset '{name}' terminated at step {result['steps']}: {result['message']}"
        )