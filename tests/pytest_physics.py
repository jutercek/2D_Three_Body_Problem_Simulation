"""
pytest version of testing version 1
"""

import numpy as np
import pytest
from three_body.physics import (
    gravitational_force,
    compute_accelerations,
    rk4_step,
    compute_total_energy,
    compute_center_of_mass,
    check_collision,
    check_escape,
    G,
    SOFTENING,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def equal_masses():
    """
    Three equal masses in a triangle
    """
    positions = np.array([
        [-10.0,  0.0],
        [ 10.0,  0.0],
        [  0.0, 10.0],
    ])
    velocities = np.array([
        [ 0.0,  0.5],
        [ 0.0, -0.5],
        [ 0.5,  0.0],
    ])
    masses = np.array([1.0, 1.0, 1.0])
    return positions, velocities, masses


@pytest.fixture
def collinear_horizontal():
    """
    Two bodies separated by 3 units and unit masses
    """
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([3.0, 0.0])
    return pos1, pos2


# ---------------------------------------------------------------------------
# gravitational_force
# ---------------------------------------------------------------------------

class TestGravitationalForce:

    def test_force_points_toward_attractor(self, collinear_horizontal):
        """
        Force on body 1 must point toward body 2
        For a body to the right, the x-component must be positive
        and the y-component zero
        """
        pos1, pos2 = collinear_horizontal
        force = gravitational_force(pos1, pos2, mass1=1.0, mass2=1.0)

        assert force[0] > 0
        assert abs(force[1]) < 1e-10

    def test_newtons_third_law(self):
        """
        Force on body A from body B must be equal and opposite to
        force on body B from body A
        Verified for an arbitrary configuration with unequal masses
        """
        pos1 = np.array([0.0, 0.0])
        pos2 = np.array([3.0, 4.0])

        f12 = gravitational_force(pos1, pos2, mass1=2.0, mass2=5.0)
        f21 = gravitational_force(pos2, pos1, mass2=2.0, mass1=5.0)

        assert np.allclose(f12, -f21, atol=1e-10)

    def test_force_scales_with_mass(self, collinear_horizontal):
        """
        Doubling one mass must double the force magnitude
        This directly tests F proportional to m1*m2
        """
        pos1, pos2 = collinear_horizontal
        f1 = gravitational_force(pos1, pos2, mass1=1.0, mass2=1.0)
        f2 = gravitational_force(pos1, pos2, mass1=2.0, mass2=1.0)

        assert np.allclose(np.linalg.norm(f2), 2 * np.linalg.norm(f1), rtol=1e-10)

    def test_force_finite_at_zero_separation(self):
        """
        Force must remain finite when two bodies occupy the same position
        Softening prevents division by zero
        """
        pos = np.array([0.0, 0.0])
        force = gravitational_force(pos, pos, mass1=1.0, mass2=1.0)

        assert np.all(np.isfinite(force))

    def test_force_decreases_with_distance(self):
        """
        Force magnitude must decrease as separation increases
        Verified by comparing force at r=3 vs r=6
        """
        pos1 = np.array([0.0, 0.0])
        pos_near = np.array([3.0, 0.0])
        pos_far  = np.array([6.0, 0.0])

        f_near = np.linalg.norm(gravitational_force(pos1, pos_near, 1.0, 1.0))
        f_far  = np.linalg.norm(gravitational_force(pos1, pos_far,  1.0, 1.0))

        assert f_near > f_far