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
