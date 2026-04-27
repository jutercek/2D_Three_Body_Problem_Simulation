"""
pytest for simulation.py
"""

import numpy as np
import pytest
from three_body.simulation import Body, run_simulation, get_preset

# ----- Fix -----
"""
These functions are repeatedly called in multiple tests
Writen here as to not repeat code later
"""

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
        Body(1.0, [ 10.0, 0.0], [ 0.0, 0.0], "Body 2"),
        Body(1.0, [-10.0, 0.0], [50.0, 0.0], "Body 3"),
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


